from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from textwrap import dedent
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from campus_agent.agent import answer_question
from campus_agent.llm import load_local_env
from campus_agent.models import RetrievalResult
from campus_agent.tools import CommunitySearchError
from campus_agent.tools import CommunitySearchResult
from campus_agent.tools import build_shuiyuan_search_tool


@dataclass(slots=True)
class WebAppDefaults:
    api_base: str
    model: str
    answer_backend: str
    community_base_url: str
    llm_configured: bool


@dataclass(slots=True)
class UserApiKeyAuthorization:
    client_id: str
    application_name: str
    nonce: str
    scopes: list[str]
    public_key_pem: str
    private_key_pem: bytes
    created_at: float
    base_url: str


@dataclass(slots=True)
class UserApiKeyGrant:
    key: str
    nonce: str
    push: bool
    api: int


@dataclass(slots=True)
class AnswerJob:
    id: str
    payload: dict[str, Any]
    status: str
    step: str
    message: str
    started_at: float
    updated_at: float
    finished_at: float | None = None
    response: dict[str, Any] | None = None
    error: str = ""


class AnswerJobManager:
    def __init__(self, *, default_answer_backend: str) -> None:
        self._default_answer_backend = default_answer_backend
        self._lock = threading.Lock()
        self._jobs: dict[str, AnswerJob] = {}

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        job = AnswerJob(
            id=str(uuid.uuid4()),
            payload=payload,
            status="running",
            step="queued",
            message="请求已提交，准备开始。",
            started_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run_job, args=(job.id,), daemon=True)
        thread.start()
        return self.status(job.id)

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("unknown answer job")
            elapsed = max(0.0, (job.finished_at or time.time()) - job.started_at)
            payload = {
                "job_id": job.id,
                "status": job.status,
                "step": job.step,
                "message": job.message,
                "elapsed_seconds": round(elapsed, 2),
            }
            if job.status == "completed" and job.response is not None:
                payload["response"] = job.response
            if job.status == "error":
                payload["error"] = job.error or job.message
            return payload

    def _run_job(self, job_id: str) -> None:
        try:
            response = build_answer_response(
                self._jobs[job_id].payload,
                default_answer_backend=self._default_answer_backend,
                progress_callback=lambda step, message: self._update(job_id, step=step, message=message),
            )
        except Exception as exc:
            self._complete(job_id, error=str(exc))
            return
        self._complete(job_id, response=response)

    def _update(self, job_id: str, *, step: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.step = step
            job.message = message
            job.updated_at = time.time()

    def _complete(
        self,
        job_id: str,
        *,
        response: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.finished_at = time.time()
            job.updated_at = job.finished_at
            if error:
                job.status = "error"
                job.step = "error"
                job.message = error
                job.error = error
                return
            job.status = "completed"
            job.step = "completed"
            job.message = "回答已生成。"
            job.response = response


class CommunityAuthManager:
    def __init__(self) -> None:
        self._states: dict[str, UserApiKeyAuthorization] = {}

    def start(
        self,
        *,
        base_url: str,
        application_name: str,
        scopes: list[str],
        client_id: str = "",
    ) -> dict[str, Any]:
        _validate_community_base_url(base_url)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        public_key = private_key.public_key()
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        resolved_client_id = client_id.strip() or str(uuid.uuid4())
        nonce = secrets.token_urlsafe(32)
        state = UserApiKeyAuthorization(
            client_id=resolved_client_id,
            application_name=application_name.strip() or "Campus Agent Studio",
            nonce=nonce,
            scopes=["read"],
            public_key_pem=public_key_pem,
            private_key_pem=private_key_pem,
            created_at=time.time(),
            base_url=base_url.rstrip("/"),
        )
        self._states[resolved_client_id] = state
        params = {
            "application_name": state.application_name,
            "client_id": state.client_id,
            "scopes": ",".join(state.scopes),
            "public_key": state.public_key_pem,
            "nonce": state.nonce,
        }
        query = "&".join(f"{key}={quote(value)}" for key, value in params.items())
        return {
            "client_id": state.client_id,
            "nonce": state.nonce,
            "scopes": state.scopes,
            "auth_url": f"{state.base_url}/user-api-key/new?{query}",
        }

    def complete(self, *, client_id: str, encrypted_payload: str) -> dict[str, Any]:
        state = self._states.get(client_id.strip())
        if state is None:
            raise ValueError("unknown or expired Shuiyuan auth client_id")
        private_key = serialization.load_pem_private_key(state.private_key_pem, password=None)
        decrypted = private_key.decrypt(
            base64.b64decode(encrypted_payload.strip()),
            padding.PKCS1v15(),
        )
        body = json.loads(decrypted.decode("utf-8"))
        grant = UserApiKeyGrant(
            key=str(body["key"]),
            nonce=str(body["nonce"]),
            push=bool(body.get("push", False)),
            api=int(body.get("api", 0)),
        )
        if grant.nonce != state.nonce:
            raise ValueError("nonce mismatch while completing Shuiyuan authorization")
        self._states.pop(state.client_id, None)
        return {
            "client_id": state.client_id,
            "user_api_key": grant.key,
            "scopes": state.scopes,
            "push": grant.push,
            "api": grant.api,
        }


def run_web_app(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    default_answer_backend: str = "llm",
) -> None:
    load_local_env()

    defaults = _resolve_defaults(default_answer_backend)
    auth_manager = CommunityAuthManager()
    job_manager = AnswerJobManager(default_answer_backend=defaults.answer_backend)
    handler = _build_handler(defaults, auth_manager, job_manager)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"web_ui=http://{host}:{port} backend={defaults.answer_backend}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down web server")
    finally:
        server.server_close()


def build_answer_response(
    payload: dict[str, Any],
    *,
    default_answer_backend: str = "llm",
    progress_callback=None,
) -> dict[str, Any]:
    question = _coerce_text(payload.get("question"))
    if not question:
        raise ValueError("question is required")

    top_k = _parse_positive_int(payload.get("top_k"), default=5)
    question_type = ""
    retrieval_mode = "hybrid"
    answer_backend = default_answer_backend
    use_query_rewriting = True
    use_rerank = True
    use_body_rag = _parse_bool(payload.get("use_body_rag"), default=True)
    llm_model = _coerce_text(payload.get("llm_model")) or None
    llm_api_key = _coerce_text(payload.get("llm_api_key")) or None
    llm_api_base = None
    llm_timeout_seconds = _parse_optional_int(payload.get("llm_timeout_seconds"))

    community_base_url = _coerce_text(payload.get("community_base_url")) or "https://shuiyuan.sjtu.edu.cn"
    _validate_community_base_url(community_base_url)
    community_user_api_key = _coerce_text(payload.get("community_user_api_key"))
    community_user_api_client_id = _coerce_text(payload.get("community_user_api_client_id"))
    community_timeout_seconds = _parse_optional_int(payload.get("community_timeout_seconds")) or 15

    if not community_user_api_key:
        raise ValueError("Shuiyuan User-Api-Key is required because answers use Shuiyuan search only")
    community_tool = build_shuiyuan_search_tool(
        base_url=community_base_url,
        user_api_key=community_user_api_key,
        user_api_client_id=community_user_api_client_id,
        timeout_seconds=community_timeout_seconds,
        progress_callback=(
            lambda message: progress_callback("searching", message)
            if progress_callback is not None
            else None
        ),
    )

    try:
        output = answer_question(
            question,
            question_type=question_type,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_query_rewriting=use_query_rewriting,
            use_rerank=use_rerank,
            use_body_rag=use_body_rag,
            community_tool=community_tool,
            answer_backend=answer_backend,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_timeout_seconds=llm_timeout_seconds,
            progress_callback=progress_callback,
        )
    except CommunitySearchError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "request": {
            "question": question,
            "top_k": top_k,
            "answer_backend": answer_backend,
            "use_body_rag": use_body_rag,
            "llm_model": llm_model,
            "llm_api_key_supplied": bool(llm_api_key),
            "llm_timeout_seconds": llm_timeout_seconds,
            "community_base_url": community_base_url,
            "community_user_api_key_supplied": bool(community_user_api_key),
            "community_search_enabled": bool(community_tool),
            "community_user_api_client_id": community_user_api_client_id,
            "community_timeout_seconds": community_timeout_seconds,
            "query_rewrite_backend": output.get("query_rewrite_backend", ""),
            "body_rag_used": bool(output.get("body_rag_used")),
            "observed_wait_seconds": int(output.get("observed_wait_seconds", 0) or 0),
            "llm_call_count": int(output.get("llm_call_count", 0) or 0),
            "search_request_count": int(output.get("search_request_count", 0) or 0),
        },
        "queries": output["queries"],
        "search_plan": output.get("search_plan", {}),
        "question_shape": output.get("question_shape", ""),
        "topic_scores": output.get("topic_scores", []),
        "answer_contract": output.get("answer_contract", {}),
        "query_details": output.get("query_details", []),
        "selected_query_details": output.get("selected_query_details", []),
        "rejected_query_details": output.get("rejected_query_details", []),
        "search_batches": output.get("search_batches", []),
        "coverage_assessments": output.get("coverage_assessments", []),
        "expanded_topics": output.get("expanded_topics", []),
        "structured_evidence": output.get("structured_evidence", {}),
        "evidence_ledger": output.get("evidence_ledger", {}),
        "answer": output["answer"],
        "results": [_serialize_result(result) for result in output["results"]],
        "community_results": [_serialize_community_result(item) for item in output["community_results"]],
    }


def build_community_auth_start_response(
    auth_manager: CommunityAuthManager,
    payload: dict[str, Any],
    *,
    default_base_url: str,
) -> dict[str, Any]:
    base_url = _coerce_text(payload.get("community_base_url")) or default_base_url
    _validate_community_base_url(base_url)
    application_name = _coerce_text(payload.get("application_name")) or "Campus Agent Studio"
    client_id = _coerce_text(payload.get("client_id"))
    return auth_manager.start(
        base_url=base_url,
        application_name=application_name,
        scopes=["read"],
        client_id=client_id,
    )


def build_community_auth_complete_response(
    auth_manager: CommunityAuthManager,
    payload: dict[str, Any],
) -> dict[str, Any]:
    client_id = _coerce_text(payload.get("client_id"))
    encrypted_payload = _coerce_text(payload.get("encrypted_payload"))
    if not client_id:
        raise ValueError("client_id is required to complete Shuiyuan authorization")
    if not encrypted_payload:
        raise ValueError("encrypted_payload is required to complete Shuiyuan authorization")
    return auth_manager.complete(client_id=client_id, encrypted_payload=encrypted_payload)


def _resolve_defaults(default_answer_backend: str) -> WebAppDefaults:
    backend = default_answer_backend.strip() or "llm"
    if backend not in {"extractive", "llm"}:
        raise ValueError(f"invalid default answer backend: {backend}")
    return WebAppDefaults(
        api_base=os.environ.get("SJTU_LLM_API_BASE", "https://models.sjtu.edu.cn/api/v1").strip(),
        model=os.environ.get("SJTU_LLM_MODEL", "deepseek-chat").strip(),
        answer_backend=backend,
        community_base_url=os.environ.get("SHUIYUAN_BASE_URL", "https://shuiyuan.sjtu.edu.cn").strip(),
        llm_configured=bool(os.environ.get("SJTU_LLM_API_KEY", "").strip()),
    )


def _build_handler(
    defaults: WebAppDefaults,
    auth_manager: CommunityAuthManager,
    job_manager: AnswerJobManager,
) -> type[BaseHTTPRequestHandler]:
    homepage = _render_homepage(defaults=defaults)

    class CampusAgentWebHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(homepage)
                return
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "default_answer_backend": defaults.answer_backend,
                        "default_api_base": defaults.api_base,
                        "default_model": defaults.model,
                        "llm_configured": defaults.llm_configured,
                        "default_community_base_url": defaults.community_base_url,
                        "community_auth_flow": "discourse-user-api-key",
                        "answer_evidence": "shuiyuan-only",
                    }
                )
                return
            if parsed.path == "/api/answer/status":
                query = parse_qs(parsed.query)
                job_id = _coerce_text((query.get("job_id") or [""])[0])
                if not job_id:
                    self._send_json({"error": "job_id is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    self._send_json(job_manager.status(job_id))
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
                if parsed.path == "/api/answer":
                    response = build_answer_response(
                        payload,
                        default_answer_backend=defaults.answer_backend,
                    )
                elif parsed.path == "/api/answer/start":
                    response = job_manager.start(payload)
                elif parsed.path == "/api/community/auth/start":
                    response = build_community_auth_start_response(
                        auth_manager,
                        payload,
                        default_base_url=defaults.community_base_url,
                    )
                elif parsed.path == "/api/community/auth/complete":
                    response = build_community_auth_complete_response(auth_manager, payload)
                else:
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                    return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # pragma: no cover
                self._send_json(
                    {"error": f"request failed: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_json(response)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON body: {exc}") from exc
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            return body

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, body: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return CampusAgentWebHandler


def _render_homepage(
    *,
    defaults: WebAppDefaults,
) -> str:
    config = json.dumps(
        {
            "default_answer_backend": defaults.answer_backend,
            "default_api_base": defaults.api_base,
            "default_model": defaults.model,
            "llm_configured": defaults.llm_configured,
            "default_community_base_url": defaults.community_base_url,
        },
        ensure_ascii=False,
    )
    return dedent(
        """
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Campus Agent Studio</title>
          <style>
            :root {
              --bg: #0a1120;
              --bg-2: #121b31;
              --panel: rgba(10, 17, 31, 0.9);
              --panel-2: rgba(14, 22, 38, 0.92);
              --border: rgba(148, 163, 184, 0.16);
              --text: #e8eef9;
              --muted: #98a8c4;
              --accent: #6ab7ff;
              --accent-2: #4fd1b8;
              --brand: #3852a3;
              --brand-soft: rgba(56, 82, 163, 0.16);
              --shadow: 0 20px 50px rgba(0, 0, 0, 0.28);
              --radius: 20px;
              --radius-sm: 14px;
              --bubble-user: linear-gradient(135deg, rgba(106, 183, 255, 0.22), rgba(79, 209, 184, 0.18));
              --bubble-assistant: rgba(255, 255, 255, 0.03);
            }
            body[data-theme="light"] {
              --bg: #edf2fb;
              --bg-2: #f8fbff;
              --panel: rgba(255, 255, 255, 0.9);
              --panel-2: rgba(246, 249, 255, 0.96);
              --border: rgba(56, 82, 163, 0.12);
              --text: #16233f;
              --muted: #62708f;
              --accent: #3f68dc;
              --accent-2: #2aa38f;
              --brand: #3852a3;
              --brand-soft: rgba(56, 82, 163, 0.08);
              --shadow: 0 18px 48px rgba(47, 72, 129, 0.12);
              --bubble-user: linear-gradient(135deg, rgba(63, 104, 220, 0.12), rgba(42, 163, 143, 0.12));
              --bubble-assistant: rgba(255, 255, 255, 0.75);
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              min-height: 100vh;
              color: var(--text);
              background:
                radial-gradient(circle at 0% 0%, rgba(106, 183, 255, 0.14), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(79, 209, 184, 0.1), transparent 24%),
                linear-gradient(180deg, var(--bg), var(--bg-2));
              font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
              transition: background 220ms ease, color 220ms ease;
            }
            .app-shell {
              display: grid;
              grid-template-columns: 360px minmax(0, 1fr);
              min-height: 100vh;
            }
            .sidebar {
              padding: 20px;
              border-right: 1px solid var(--border);
              background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
              backdrop-filter: blur(22px);
              overflow: auto;
            }
            .main {
              min-width: 0;
              display: flex;
              flex-direction: column;
              min-height: 100vh;
            }
            .topbar {
              position: sticky;
              top: 0;
              z-index: 5;
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 12px;
              padding: 18px 24px;
              background: rgba(10, 17, 31, 0.54);
              backdrop-filter: blur(18px);
              border-bottom: 1px solid var(--border);
            }
            body[data-theme="light"] .topbar {
              background: rgba(248, 251, 255, 0.72);
            }
            .brand {
              display: flex;
              align-items: center;
              gap: 14px;
            }
            .brand-mark {
              width: 54px;
              height: 54px;
              flex: 0 0 auto;
              color: var(--brand);
            }
            .brand-copy strong {
              display: block;
              font-size: 26px;
              letter-spacing: 0.01em;
              color: var(--brand);
              line-height: 1;
            }
            .brand-copy span {
              display: block;
              margin-top: 6px;
              color: var(--muted);
              font-size: 13px;
            }
            .topbar-actions {
              display: flex;
              gap: 10px;
              align-items: center;
            }
            .page {
              width: min(980px, calc(100% - 32px));
              margin: 0 auto;
              padding: 28px 0 34px;
              display: grid;
              gap: 18px;
            }
            .card {
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: var(--radius);
              box-shadow: var(--shadow);
              backdrop-filter: blur(18px);
            }
            .panel, .chat-card, .composer-card { padding: 22px; }
            .small, .meta, .status-line, .helper-text { color: var(--muted); }
            .pill, .chip, .tag {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              padding: 8px 12px;
              border-radius: 999px;
              font-size: 13px;
            }
            .pill { background: var(--brand-soft); color: var(--brand); }
            .chip {
              background: rgba(148, 163, 184, 0.1);
              color: var(--text);
            }
            .tag {
              background: rgba(106, 183, 255, 0.12);
              color: var(--accent);
            }
            .panel h2, .panel h3, .chat-title { margin: 0; }
            .stack { display: grid; gap: 14px; }
            .field-grid, .two-col { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .label { display: block; margin-bottom: 8px; font-size: 13px; color: var(--text); }
            input, select, textarea {
              width: 100%;
              padding: 13px 14px;
              border-radius: 16px;
              border: 1px solid var(--border);
              background: var(--panel-2);
              color: var(--text);
              font: inherit;
              outline: none;
            }
            input:focus, select:focus, textarea:focus { border-color: rgba(106, 183, 255, 0.85); box-shadow: 0 0 0 3px rgba(106, 183, 255, 0.12); }
            textarea { min-height: 120px; resize: vertical; }
            .toggle {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              padding: 10px 12px;
              border-radius: 999px;
              background: rgba(148, 163, 184, 0.09);
              font-size: 13px;
              color: var(--text);
            }
            .toggle input { width: auto; accent-color: var(--accent-2); }
            .actions { display: flex; gap: 12px; flex-wrap: wrap; }
            button {
              border: 0;
              border-radius: 14px;
              padding: 12px 16px;
              font: inherit;
              font-weight: 700;
              cursor: pointer;
              transition: transform 160ms ease, opacity 160ms ease, background 180ms ease;
            }
            button:hover { transform: translateY(-1px); }
            .primary { color: white; background: linear-gradient(135deg, var(--brand), var(--accent)); box-shadow: 0 14px 28px rgba(56, 82, 163, 0.26); }
            .ghost { color: var(--text); background: rgba(148, 163, 184, 0.1); }
            .ghost.subtle { background: transparent; border: 1px solid var(--border); }
            .pill-row, .citation-list { display: flex; flex-wrap: wrap; gap: 8px; }
            .citation-list { display: grid; gap: 12px; }
            .citation { padding: 14px 16px; border-radius: 16px; border: 1px solid var(--border); background: var(--panel-2); }
            .citation .title { margin-bottom: 4px; font-weight: 700; }
            .citation a, .assistant-answer a, .msg-body a { color: var(--accent); word-break: break-all; }
            .section + .section { margin-top: 18px; }
            .sidebar-head {
              display: flex;
              align-items: center;
              justify-content: space-between;
              margin-bottom: 16px;
            }
            .sidebar-block {
              padding: 18px;
              border-radius: var(--radius);
              border: 1px solid var(--border);
              background: var(--panel);
              box-shadow: var(--shadow);
            }
            .sidebar-block + .sidebar-block {
              margin-top: 14px;
            }
            .setup-guide {
              display: grid;
              gap: 10px;
            }
            .setup-step {
              display: grid;
              grid-template-columns: 30px minmax(0, 1fr) auto;
              gap: 10px;
              align-items: center;
              padding: 12px;
              border-radius: 14px;
              border: 1px solid var(--border);
              background: var(--panel-2);
            }
            .setup-step-number {
              width: 28px;
              height: 28px;
              border-radius: 50%;
              display: grid;
              place-items: center;
              background: var(--brand-soft);
              color: var(--brand);
              font-weight: 800;
            }
            .setup-step strong, .setup-step span { display: block; }
            .setup-state {
              padding: 5px 8px;
              border-radius: 999px;
              color: var(--muted);
              background: rgba(148, 163, 184, 0.1);
              font-size: 11px;
              white-space: nowrap;
            }
            .setup-state.ready {
              color: var(--accent-2);
              background: rgba(79, 209, 184, 0.12);
            }
            .advanced-settings summary {
              cursor: pointer;
              color: var(--muted);
              font-weight: 700;
            }
            .advanced-settings[open] summary { margin-bottom: 14px; }
            .security-note {
              padding: 11px 12px;
              border-radius: 14px;
              background: rgba(79, 209, 184, 0.08);
              color: var(--muted);
              font-size: 12px;
              line-height: 1.55;
            }
            .chat-card {
              display: grid;
              gap: 16px;
            }
            .progress-card {
              border: 1px solid var(--border);
              border-radius: 18px;
              background: var(--panel-2);
              padding: 16px 18px;
              display: grid;
              gap: 12px;
            }
            .progress-head {
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 12px;
              flex-wrap: wrap;
            }
            .progress-time {
              font-weight: 800;
              color: var(--accent);
            }
            .progress-steps {
              display: grid;
              grid-template-columns: repeat(6, minmax(0, 1fr));
              gap: 10px;
            }
            .progress-step {
              padding: 10px 12px;
              border-radius: 14px;
              border: 1px solid var(--border);
              background: rgba(148, 163, 184, 0.06);
              color: var(--muted);
              font-size: 13px;
            }
            .progress-step.active {
              background: rgba(106, 183, 255, 0.12);
              color: var(--text);
              border-color: rgba(106, 183, 255, 0.35);
            }
            .progress-step.done {
              background: rgba(79, 209, 184, 0.12);
              color: var(--text);
              border-color: rgba(79, 209, 184, 0.35);
            }
            .progress-step strong {
              display: block;
              margin-bottom: 4px;
            }
            .chat-stream {
              display: grid;
              gap: 14px;
            }
            .message {
              display: flex;
              gap: 12px;
              align-items: flex-start;
            }
            .message.user {
              justify-content: flex-end;
            }
            .avatar {
              width: 36px;
              height: 36px;
              border-radius: 50%;
              display: grid;
              place-items: center;
              background: var(--brand-soft);
              color: var(--brand);
              font-weight: 800;
              flex: 0 0 auto;
            }
            .bubble {
              max-width: min(760px, 100%);
              border-radius: 20px;
              border: 1px solid var(--border);
              padding: 16px 18px;
              box-shadow: var(--shadow);
            }
            .message.user .bubble {
              background: var(--bubble-user);
            }
            .message.assistant .bubble, .message.system .bubble {
              background: var(--bubble-assistant);
            }
            .message-head {
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 10px;
              margin-bottom: 8px;
            }
            .message-head strong {
              font-size: 15px;
            }
            .msg-body {
              white-space: pre-wrap;
              line-height: 1.7;
            }
            .assistant-answer {
              white-space: pre-wrap;
              line-height: 1.76;
              font-size: 15px;
            }
            .composer-card {
              position: sticky;
              bottom: 0;
              background: linear-gradient(180deg, rgba(255,255,255,0), var(--bg-2) 28%, var(--bg-2));
              padding-top: 8px;
            }
            .composer {
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: 24px;
              padding: 16px;
              box-shadow: var(--shadow);
            }
            .composer textarea {
              min-height: 110px;
              border-radius: 18px;
            }
            .composer-actions {
              margin-top: 12px;
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 12px;
              flex-wrap: wrap;
            }
            .details-card {
              overflow: hidden;
            }
            .details-card details {
              border-top: 1px solid var(--border);
            }
            .details-card details:first-child {
              border-top: 0;
            }
            .details-card summary {
              list-style: none;
              cursor: pointer;
              padding: 16px 18px;
              font-weight: 700;
              display: flex;
              align-items: center;
              justify-content: space-between;
            }
            .details-card summary::-webkit-details-marker { display: none; }
            .details-body {
              padding: 0 18px 18px;
            }
            .mobile-settings-toggle {
              display: none;
            }
            .sidebar.hidden-mobile {
              display: block;
            }
            .status-line {
              display: none;
              padding: 11px 13px;
              border-radius: 14px;
              background: rgba(148, 163, 184, 0.08);
            }
            .status-line.ready { display: block; color: #bff7e7; }
            .status-line.working { display: block; color: #d9e6ff; }
            .status-line.error { display: block; color: #ffc0cb; }
            @media (max-width: 1180px) {
              .app-shell { grid-template-columns: 1fr; }
              .sidebar {
                position: fixed;
                inset: 0 auto 0 0;
                width: min(420px, 100vw);
                z-index: 15;
                transform: translateX(-102%);
                transition: transform 180ms ease;
                background: var(--bg-2);
              }
              .sidebar.open { transform: translateX(0); }
              .mobile-settings-toggle { display: inline-flex; }
            }
            @media (max-width: 720px) {
              .topbar, .page { width: 100%; }
              .page { padding: 16px; }
              .field-grid, .two-col, .progress-steps { grid-template-columns: 1fr; }
              .brand-copy strong { font-size: 22px; }
              .topbar { padding: 16px; }
            }
          </style>
        </head>
        <body data-theme="dark">
          <div class="app-shell">
            <aside id="sidebar" class="sidebar">
              <div class="sidebar-head">
                <div>
                  <strong>开始使用</strong>
                  <div class="helper-text">完成两项配置后即可提问</div>
                </div>
                <button id="close-sidebar-btn" class="ghost subtle mobile-settings-toggle" type="button">关闭</button>
              </div>

              <form id="ask-form" class="stack">
                <div class="sidebar-block">
                  <div class="setup-guide">
                    <div class="setup-step">
                      <span class="setup-step-number">1</span>
                      <div><strong>配置校内模型</strong><span class="helper-text">使用交大本地模型服务</span></div>
                      <span id="llm-ready-state" class="setup-state">待配置</span>
                    </div>
                    <div class="setup-step">
                      <span class="setup-step-number">2</span>
                      <div><strong>授权 Shuiyuan</strong><span class="helper-text">仅申请 read 权限</span></div>
                      <span id="community-ready-state" class="setup-state">待授权</span>
                    </div>
                    <div class="setup-step">
                      <span class="setup-step-number">3</span>
                      <div><strong>输入校园问题</strong><span class="helper-text">Agent 会规划搜索并整理证据</span></div>
                      <span class="setup-state">开始提问</span>
                    </div>
                  </div>
                </div>

                <div class="sidebar-block">
                  <h3>1. 校内模型</h3>
                  <div class="helper-text">推荐在本地 <code>.env.local</code> 配置；也可仅为当前页面临时填写。</div>
                  <div class="section">
                    <label class="label" for="api_key">模型 API Key</label>
                    <input id="api_key" type="password" autocomplete="off" placeholder="留空则使用本地 .env.local 配置" />
                  </div>
                  <div class="field-grid">
                    <div>
                      <label class="label" for="llm_model">模型名</label>
                      <input id="llm_model" type="text" />
                    </div>
                    <div>
                      <label class="label" for="llm_timeout_seconds">超时秒数</label>
                      <input id="llm_timeout_seconds" type="number" min="1" step="1" value="60" />
                    </div>
                  </div>
                  <div class="section security-note">模型请求只允许发送到配置的交大校内模型域名，页面不会保存 API Key。</div>
                </div>

                <div class="sidebar-block">
                  <h3>2. Shuiyuan 只读授权</h3>
                  <div class="helper-text">点击开始授权，在新页面确认后，将返回的加密 payload 粘贴回来。</div>
                  <div class="section">
                    <label class="label" for="community_encrypted_payload">授权返回 payload</label>
                    <textarea id="community_encrypted_payload" autocomplete="off" placeholder="粘贴 Shuiyuan 授权页返回的加密 payload"></textarea>
                  </div>
                  <div class="actions">
                    <button id="community-auth-start-btn" class="primary" type="button">开始只读授权</button>
                    <button id="community-auth-complete-btn" class="ghost" type="button">完成授权</button>
                  </div>
                  <div class="section security-note">仅申请 Discourse <code>read</code> scope。User-Api-Key 只在当前页面会话中使用，不写入浏览器存储。</div>
                  <input id="community_user_api_client_id" type="hidden" />
                  <input id="community_user_api_key" type="hidden" />
                  <input id="community_application_name" type="hidden" value="Shuiyuan Agent" />
                  <input id="community_scopes" type="hidden" value="read" />
                </div>

                <div class="sidebar-block">
                  <details class="advanced-settings">
                    <summary>高级设置</summary>
                    <div class="section">
                      <label class="toggle"><input id="use_body_rag" type="checkbox" checked />启用帖子正文证据补全</label>
                    </div>
                    <div class="section field-grid">
                      <div>
                        <label class="label" for="top_k">展示证据数</label>
                        <input id="top_k" type="number" min="1" max="20" step="1" value="5" />
                      </div>
                      <div>
                        <label class="label" for="community_timeout_seconds">社区请求超时</label>
                        <input id="community_timeout_seconds" type="number" min="1" step="1" value="15" />
                      </div>
                    </div>
                    <div class="section">
                      <label class="label" for="community_base_url">Shuiyuan Base URL</label>
                      <input id="community_base_url" type="text" readonly />
                    </div>
                  </details>
                </div>
              </form>
            </aside>

            <main class="main">
              <header class="topbar">
                <div class="brand">
                  <svg class="brand-mark" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M58 12C82 12 101 30 101 54C101 79 82 97 58 97C50 97 42 95 35 92L19 104L24 84C18 76 15 65 15 54C15 30 34 12 58 12Z" stroke="currentColor" stroke-width="7" stroke-linejoin="round"/>
                    <path d="M39 66C42 54 49 45 58 40C63 37 69 36 75 37C69 31 61 28 53 28C39 28 29 38 29 52C29 59 32 66 39 71V66Z" fill="currentColor" opacity="0.18"/>
                    <path d="M43 73C47 58 58 48 72 47C77 47 82 48 87 51C80 64 68 74 55 77C50 78 46 77 43 73Z" fill="currentColor"/>
                    <circle cx="43" cy="39" r="5" fill="currentColor"/>
                  </svg>
                  <div class="brand-copy">
                    <strong>水源Agent</strong>
                    <span>Shuiyuan community search assistant</span>
                  </div>
                </div>
                <div class="topbar-actions">
                  <button id="toggle-theme-btn" class="ghost subtle" type="button">切换主题</button>
                  <button id="open-sidebar-btn" class="ghost subtle mobile-settings-toggle" type="button">设置</button>
                </div>
              </header>

              <section class="page">
                <article class="card chat-card">
                  <div class="progress-card">
                    <div class="progress-head">
                      <div>
                        <strong>运行进度</strong>
                        <div id="progress-message" class="helper-text">等待提问。</div>
                      </div>
                      <div class="progress-time" id="progress-time">0.0s</div>
                    </div>
                    <div id="progress-steps" class="progress-steps">
                      <div class="progress-step" data-step="query_planning"><strong>1. 理解问题</strong><span>生成动态答案目标</span></div>
                      <div class="progress-step" data-step="searching"><strong>2. 分批搜索</strong><span>执行多方向 Shuiyuan 查询</span></div>
                      <div class="progress-step" data-step="evidence_audit"><strong>3. 证据审计</strong><span>判断已覆盖和缺失信息</span></div>
                      <div class="progress-step" data-step="body_rag"><strong>4. 深度补全</strong><span>按缺口展开帖子回复</span></div>
                      <div class="progress-step" data-step="fact_ledger"><strong>5. 事实整理</strong><span>只保留直接支持的事实</span></div>
                      <div class="progress-step" data-step="generating"><strong>6. 答案校验</strong><span>生成并核对最终回答</span></div>
                    </div>
                  </div>
                  <div class="chat-stream">
                    <div class="message system">
                      <div class="avatar">S</div>
                      <div class="bubble">
                        <div class="message-head">
                          <strong>Shuiyuan Agent</strong>
                          <span class="small">只读社区搜索</span>
                        </div>
                        <div class="msg-body">先在设置中完成校内模型配置和 Shuiyuan 只读授权。提问后，系统会规划多组短查询、审计证据并按需阅读高相关帖子正文。</div>
                      </div>
                    </div>

                    <div class="message user">
                      <div class="bubble">
                        <div class="message-head">
                          <strong>你的问题</strong>
                        </div>
                        <div id="question-preview" class="msg-body">还没有输入问题。</div>
                      </div>
                    </div>

                    <div class="message assistant">
                      <div class="avatar">A</div>
                      <div class="bubble">
                        <div class="message-head">
                          <strong>回答</strong>
                          <div id="request-pills" class="pill-row"></div>
                        </div>
                        <div id="answer-box" class="assistant-answer">回答会显示在这里。</div>
                      </div>
                    </div>
                  </div>
                  <div id="status-line" class="status-line">等待提问。</div>
                </article>

                <article class="card details-card">
                  <details open>
                    <summary>搜索计划 <span id="plan-intent" class="small">尚未生成 intent。</span></summary>
                    <div class="details-body">
                      <div class="section">
                        <div class="small">桥接概念</div>
                        <div id="bridges" class="pill-row"></div>
                      </div>
                      <div class="section">
                        <div class="small">候选 Queries</div>
                        <div id="candidate-queries" class="pill-row"></div>
                      </div>
                      <div class="section">
                        <div class="small">实际执行 Queries</div>
                        <div id="queries" class="pill-row"></div>
                      </div>
                      <div class="section">
                        <div class="small">LLM 淘汰的低价值 Queries</div>
                        <div id="rejected-queries" class="citation-list"></div>
                      </div>
                    </div>
                  </details>
                  <details open>
                    <summary>动态答案目标</summary>
                    <div class="details-body">
                      <div id="answer-contract" class="citation-list"></div>
                    </div>
                  </details>
                  <details>
                    <summary>证据覆盖与补查过程</summary>
                    <div class="details-body">
                      <div id="coverage-assessments" class="citation-list"></div>
                    </div>
                  </details>
                  <details>
                    <summary>最终事实账本</summary>
                    <div class="details-body">
                      <div id="evidence-ledger" class="citation-list"></div>
                    </div>
                  </details>
                  <details>
                    <summary>采用的证据片段</summary>
                    <div class="details-body">
                      <div id="results" class="citation-list"></div>
                    </div>
                  </details>
                  <details>
                    <summary>Shuiyuan 社区帖子</summary>
                    <div class="details-body">
                      <div id="community-results" class="citation-list"></div>
                    </div>
                  </details>
                </article>

                <section class="composer-card">
                  <div class="composer">
                    <label class="label" for="question">问题</label>
                    <textarea id="question" placeholder="例如：校车怎么坐？宿舍网络怎么申请？晚上七点后还有哪个食堂开着？"></textarea>
                    <div class="composer-actions">
                      <div class="helper-text">按 Ctrl/⌘ + Enter 提交。Agent 会分批搜索，并按信息缺口补全证据。</div>
                      <div class="actions">
                        <button id="copy-btn" class="ghost" type="button">复制回答</button>
                        <button id="submit-btn" class="primary" type="submit" form="ask-form">生成答案</button>
                      </div>
                    </div>
                  </div>
                </section>
              </section>
            </main>
          </div>

          <script>
            const APP_CONFIG = __APP_CONFIG__;
            const state = {
              lastAnswer: "",
              lastAuthUrl: "",
              theme: localStorage.getItem("shuiyuan-agent-theme") || "dark",
              progressTimer: null,
              progressStartedAt: 0,
              currentJobId: "",
            };

            const elements = {
              form: document.getElementById("ask-form"),
              question: document.getElementById("question"),
              questionPreview: document.getElementById("question-preview"),
              topK: document.getElementById("top_k"),
              useBodyRag: document.getElementById("use_body_rag"),
              apiKey: document.getElementById("api_key"),
              llmModel: document.getElementById("llm_model"),
              llmTimeout: document.getElementById("llm_timeout_seconds"),
              communityBaseUrl: document.getElementById("community_base_url"),
              communityTimeout: document.getElementById("community_timeout_seconds"),
              communityApplicationName: document.getElementById("community_application_name"),
              communityScopes: document.getElementById("community_scopes"),
              communityClientId: document.getElementById("community_user_api_client_id"),
              communityUserApiKey: document.getElementById("community_user_api_key"),
              communityEncryptedPayload: document.getElementById("community_encrypted_payload"),
              communityAuthStartBtn: document.getElementById("community-auth-start-btn"),
              communityAuthCompleteBtn: document.getElementById("community-auth-complete-btn"),
              openSidebarBtn: document.getElementById("open-sidebar-btn"),
              closeSidebarBtn: document.getElementById("close-sidebar-btn"),
              toggleThemeBtn: document.getElementById("toggle-theme-btn"),
              sidebar: document.getElementById("sidebar"),
              submitBtn: document.getElementById("submit-btn"),
              copyBtn: document.getElementById("copy-btn"),
              statusLine: document.getElementById("status-line"),
              progressMessage: document.getElementById("progress-message"),
              progressTime: document.getElementById("progress-time"),
              progressSteps: Array.from(document.querySelectorAll(".progress-step")),
              answerBox: document.getElementById("answer-box"),
              planIntent: document.getElementById("plan-intent"),
              bridges: document.getElementById("bridges"),
              candidateQueries: document.getElementById("candidate-queries"),
              queries: document.getElementById("queries"),
              rejectedQueries: document.getElementById("rejected-queries"),
              answerContract: document.getElementById("answer-contract"),
              coverageAssessments: document.getElementById("coverage-assessments"),
              evidenceLedger: document.getElementById("evidence-ledger"),
              results: document.getElementById("results"),
              communityResults: document.getElementById("community-results"),
              requestPills: document.getElementById("request-pills"),
              llmReadyState: document.getElementById("llm-ready-state"),
              communityReadyState: document.getElementById("community-ready-state"),
            };

            document.body.dataset.theme = state.theme;
            elements.llmModel.value = APP_CONFIG.default_model;
            elements.communityBaseUrl.value = APP_CONFIG.default_community_base_url;

            function updateSetupState() {
              const llmReady = APP_CONFIG.llm_configured || Boolean(elements.apiKey.value.trim());
              const communityReady = Boolean(elements.communityUserApiKey.value.trim());
              elements.llmReadyState.textContent = llmReady ? "已就绪" : "待配置";
              elements.llmReadyState.classList.toggle("ready", llmReady);
              elements.communityReadyState.textContent = communityReady ? "已授权" : "待授权";
              elements.communityReadyState.classList.toggle("ready", communityReady);
            }

            function setStatus(message, kind = "") {
              if (!message || !kind) {
                elements.statusLine.textContent = "";
                elements.statusLine.className = "status-line";
                return;
              }
              elements.statusLine.textContent = message;
              elements.statusLine.className = `status-line ${kind}`;
            }

            function setProgressState(step, message, elapsedSeconds = null, status = "running") {
              elements.progressMessage.textContent = message || "等待提问。";
              if (elapsedSeconds !== null) {
                elements.progressTime.textContent = `${Number(elapsedSeconds).toFixed(1)}s`;
              }
              const order = ["query_planning", "searching", "evidence_audit", "body_rag", "fact_ledger", "generating"];
              const currentIndex = order.indexOf(step);
              elements.progressSteps.forEach((node, index) => {
                const stepName = node.dataset.step;
                node.classList.remove("active", "done");
                if (status === "completed") {
                  node.classList.add("done");
                  return;
                }
                if (currentIndex > index) {
                  node.classList.add("done");
                } else if (stepName === step) {
                  node.classList.add("active");
                }
              });
            }

            function startProgressClock() {
              stopProgressClock();
              state.progressStartedAt = Date.now();
              state.progressTimer = window.setInterval(() => {
                const elapsed = (Date.now() - state.progressStartedAt) / 1000;
                elements.progressTime.textContent = `${elapsed.toFixed(1)}s`;
              }, 100);
            }

            function stopProgressClock() {
              if (state.progressTimer !== null) {
                clearInterval(state.progressTimer);
                state.progressTimer = null;
              }
            }

            function updateThemeButtonLabel() {
              elements.toggleThemeBtn.textContent = document.body.dataset.theme === "light" ? "切换暗色" : "切换明亮";
            }

            function setTheme(theme) {
              document.body.dataset.theme = theme;
              state.theme = theme;
              localStorage.setItem("shuiyuan-agent-theme", theme);
              updateThemeButtonLabel();
            }

            function setSidebarOpen(open) {
              elements.sidebar.classList.toggle("open", open);
            }

            function clearNode(node) {
              while (node.firstChild) node.removeChild(node.firstChild);
            }

            function escapeHtml(text) {
              return String(text)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#39;");
            }

            function linkifyAndFormat(text) {
              const escaped = escapeHtml(text);
              const linked = escaped.replace(
                new RegExp("(https?://[^\\\\s<]+)", "g"),
                '<a href="$1" target="_blank" rel="noreferrer">$1</a>'
              );
              return linked
                .replace(/[*][*]([^*]+)[*][*]/g, "<strong>$1</strong>")
                .replaceAll("\\n", "<br>");
            }

            function createPill(text) {
              const pill = document.createElement("span");
              pill.className = "pill";
              pill.textContent = text;
              return pill;
            }

            function renderPills(container, items) {
              clearNode(container);
              if (!items.length) {
                container.appendChild(createPill("无"));
                return;
              }
              items.forEach((item) => container.appendChild(createPill(item)));
            }

            function renderCards(container, items, emptyText, isCommunity = false) {
              clearNode(container);
              if (!items.length) {
                const empty = document.createElement("div");
                empty.className = "small";
                empty.textContent = emptyText;
                container.appendChild(empty);
                return;
              }
              items.forEach((item) => {
                const card = document.createElement("article");
                card.className = "citation";
                const title = document.createElement("div");
                title.className = "title";
                title.textContent = item.title || item.chunk_id || item.url || "未命名证据";
                card.appendChild(title);
                const meta = document.createElement("div");
                meta.className = "meta";
                meta.textContent = isCommunity
                  ? `相关性 ${Number(item.relevance_score || 0).toFixed(3)} · Shuiyuan 社区 · ${item.body_loaded ? "正文已抓取" : "仅搜索摘要"} · ${item.updated_at || item.created_at || "时间未知"}`
                  : `score ${Number(item.score || 0).toFixed(3)} · 正文命中片段`;
                card.appendChild(meta);
                const linkTarget = item.uri || item.url;
                if (linkTarget) {
                  const link = document.createElement("a");
                  link.href = linkTarget;
                  link.target = "_blank";
                  link.rel = "noreferrer";
                  link.textContent = linkTarget;
                  card.appendChild(link);
                }
                const snippet = document.createElement("div");
                snippet.className = "small";
                snippet.textContent = (item.highlights && item.highlights[0]) || item.snippet || "无摘要";
                card.appendChild(snippet);
                container.appendChild(card);
              });
            }

            function renderStructured(container, value, emptyText) {
              clearNode(container);
              if (!value || (Array.isArray(value) && !value.length) || (typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length)) {
                const empty = document.createElement("div");
                empty.className = "small";
                empty.textContent = emptyText;
                container.appendChild(empty);
                return;
              }
              const items = Array.isArray(value) ? value : [value];
              items.forEach((item) => {
                const card = document.createElement("article");
                card.className = "citation";
                if (typeof item !== "object" || item === null) {
                  card.textContent = String(item);
                  container.appendChild(card);
                  return;
                }
                Object.entries(item).forEach(([key, rawValue]) => {
                  const row = document.createElement("div");
                  row.className = "small";
                  const label = document.createElement("strong");
                  label.textContent = `${key}: `;
                  row.appendChild(label);
                  const valueText = Array.isArray(rawValue)
                    ? rawValue.map((entry) => typeof entry === "object" ? JSON.stringify(entry, null, 0) : String(entry)).join("；")
                    : typeof rawValue === "object" && rawValue !== null
                      ? JSON.stringify(rawValue, null, 0)
                      : String(rawValue ?? "");
                  row.appendChild(document.createTextNode(valueText));
                  card.appendChild(row);
                });
                container.appendChild(card);
              });
            }

            function renderRequestPills(request) {
              const items = [
                "LLM + Shuiyuan",
              ];
              if (request.query_rewrite_backend) items.push(`Query planning: ${request.query_rewrite_backend}`);
              if (request.executed_query_count) items.push(`Executed queries: ${request.executed_query_count}`);
              if (request.search_request_count) items.push(`Requests: ${request.search_request_count}`);
              items.push(request.use_body_rag === false ? "仅搜索摘要" : "深度证据补全");
              if (request.community_user_api_key_supplied) items.push("Shuiyuan ready");
              if (!request.community_search_enabled) items.push("Shuiyuan not configured");
              renderPills(elements.requestPills, items);
            }

            async function postJson(url, payload) {
              const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              const data = await response.json();
              if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
              return data;
            }

            async function pollAnswerJob(jobId) {
              while (true) {
                const response = await fetch(`/api/answer/status?job_id=${encodeURIComponent(jobId)}`);
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
                setProgressState(data.step, data.message, data.elapsed_seconds, data.status);
                if (data.status === "completed") {
                  return data.response;
                }
                if (data.status === "error") {
                  throw new Error(data.error || data.message || "请求失败");
                }
                await new Promise((resolve) => setTimeout(resolve, 400));
              }
            }

            async function loadHealth() {
              try {
                const response = await fetch("/api/health");
                const data = await response.json();
                APP_CONFIG.llm_configured = Boolean(data.llm_configured);
                updateSetupState();
                setStatus("", "ready");
              } catch (error) {
                setStatus("健康检查失败，但仍可尝试提问。", "error");
              }
            }

            elements.toggleThemeBtn.addEventListener("click", () => {
              setTheme(document.body.dataset.theme === "light" ? "dark" : "light");
            });

            elements.openSidebarBtn.addEventListener("click", () => setSidebarOpen(true));
            elements.closeSidebarBtn.addEventListener("click", () => setSidebarOpen(false));

            elements.question.addEventListener("input", () => {
              elements.questionPreview.textContent = elements.question.value.trim() || "还没有输入问题。";
            });
            elements.question.addEventListener("keydown", (event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                elements.form.requestSubmit();
              }
            });
            elements.apiKey.addEventListener("input", updateSetupState);

            elements.communityAuthStartBtn.addEventListener("click", async () => {
              try {
                setStatus("正在生成 Shuiyuan 授权链接...", "working");
                const data = await postJson("/api/community/auth/start", {
                  community_base_url: elements.communityBaseUrl.value.trim(),
                  application_name: elements.communityApplicationName.value.trim(),
                  client_id: elements.communityClientId.value.trim(),
                });
                state.lastAuthUrl = data.auth_url;
                elements.communityClientId.value = data.client_id || "";
                window.open(data.auth_url, "_blank", "noopener,noreferrer");
                setStatus("已打开 Shuiyuan 授权页。授权后把返回 payload 粘贴到下方，再点“完成授权”。", "ready");
              } catch (error) {
                setStatus(error.message || String(error), "error");
              }
            });

            elements.communityAuthCompleteBtn.addEventListener("click", async () => {
              try {
                setStatus("正在解密 Shuiyuan 授权 payload...", "working");
                const data = await postJson("/api/community/auth/complete", {
                  client_id: elements.communityClientId.value.trim(),
                  encrypted_payload: elements.communityEncryptedPayload.value.trim(),
                });
                elements.communityUserApiKey.value = data.user_api_key || "";
                elements.communityClientId.value = data.client_id || elements.communityClientId.value;
                elements.communityEncryptedPayload.value = "";
                updateSetupState();
                setStatus("Shuiyuan User-Api-Key 已就绪。现在可以直接提问并带上社区搜索。", "ready");
              } catch (error) {
                setStatus(error.message || String(error), "error");
              }
            });

            elements.copyBtn.addEventListener("click", async () => {
              if (!state.lastAnswer) {
                setStatus("当前还没有可复制的回答。", "error");
                return;
              }
              try {
                await navigator.clipboard.writeText(state.lastAnswer);
                setStatus("回答已复制到剪贴板。", "ready");
              } catch (error) {
                setStatus("复制失败，请手动选择文本。", "error");
              }
            });

            elements.form.addEventListener("submit", async (event) => {
              event.preventDefault();
              const question = elements.question.value.trim();
              if (!question) {
                setStatus("请先输入问题。", "error");
                return;
              }
              if (!APP_CONFIG.llm_configured && !elements.apiKey.value.trim()) {
                setStatus("请先配置校内模型 API Key。", "error");
                setSidebarOpen(true);
                return;
              }
              if (!elements.communityUserApiKey.value.trim()) {
                setStatus("请先完成 Shuiyuan 只读授权。", "error");
                setSidebarOpen(true);
                return;
              }
              const payload = {
                question,
                top_k: Number(elements.topK.value) || 5,
                use_body_rag: elements.useBodyRag.checked,
                llm_model: elements.llmModel.value.trim(),
                llm_api_key: elements.apiKey.value,
                llm_timeout_seconds: Number(elements.llmTimeout.value) || 60,
                community_base_url: elements.communityBaseUrl.value.trim(),
                community_user_api_key: elements.communityUserApiKey.value,
                community_user_api_client_id: elements.communityClientId.value.trim(),
                community_timeout_seconds: Number(elements.communityTimeout.value) || 15,
              };

              elements.submitBtn.disabled = true;
              setStatus("正在处理中...", "working");
              elements.questionPreview.textContent = question;
              setProgressState("query_planning", "请求已提交，准备开始。", 0, "running");
              startProgressClock();
              try {
                const started = await postJson("/api/answer/start", payload);
                state.currentJobId = started.job_id || "";
                const data = await pollAnswerJob(state.currentJobId);
                state.lastAnswer = data.answer || "";
                elements.answerBox.innerHTML = linkifyAndFormat(data.answer || "未生成回答。");
                elements.planIntent.textContent = (data.search_plan && data.search_plan.intent) || "未生成 intent。";
                renderPills(elements.bridges, (data.search_plan && data.search_plan.bridges) || []);
                renderPills(elements.candidateQueries, (data.search_plan && data.search_plan.candidate_queries) || []);
                renderPills(elements.queries, data.queries || []);
                renderStructured(elements.rejectedQueries, data.rejected_query_details || [], "没有淘汰 query。");
                renderStructured(elements.answerContract, data.answer_contract || {}, "未生成动态答案目标。");
                renderStructured(elements.coverageAssessments, data.coverage_assessments || [], "未执行证据覆盖审计。");
                renderStructured(elements.evidenceLedger, data.evidence_ledger || {}, "未生成事实账本。");
                renderCards(elements.results, data.results || [], "没有命中帖子正文片段。", false);
                renderCards(elements.communityResults, data.community_results || [], "没有找到相关 Shuiyuan 帖子。", true);
                const request = data.request || payload;
                request.executed_query_count = (data.queries || []).length;
                renderRequestPills(request);
                setStatus("回答已生成。", "ready");
                setProgressState("generating", "回答已生成。", null, "completed");
                setSidebarOpen(false);
              } catch (error) {
                setStatus(error.message || String(error), "error");
                setProgressState("generating", error.message || String(error), null, "error");
              } finally {
                stopProgressClock();
                elements.submitBtn.disabled = false;
              }
            });

            setTheme(state.theme);
            updateSetupState();
            loadHealth();
            setProgressState("", "等待提问。", 0, "idle");
            renderRequestPills({
              use_body_rag: true,
              llm_api_key_supplied: false,
              community_user_api_key_supplied: false,
              community_user_api_client_id: "",
            });
          </script>
        </body>
        </html>
        """
    ).replace("__APP_CONFIG__", config)


def _serialize_result(result: RetrievalResult) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "title": chunk.metadata.get("document_title", chunk.document_id),
        "uri": chunk.metadata.get("post_url") or chunk.metadata.get("topic_url") or chunk.metadata.get("document_uri", ""),
        "score": result.score,
        "highlights": result.highlights,
        "metadata": {
            "published_at": chunk.metadata.get("published_at", ""),
            "updated_at": chunk.metadata.get("updated_at", ""),
            "evidence_origin": chunk.metadata.get("evidence_origin", ""),
            "post_number": chunk.metadata.get("post_number", ""),
        },
    }


def _serialize_community_result(result: CommunitySearchResult) -> dict[str, Any]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "created_at": result.created_at,
        "updated_at": result.updated_at,
        "reply_count": result.reply_count,
        "like_count": result.like_count,
        "relevance_score": result.relevance_score,
        "body_loaded": result.body_loaded,
        "tags": result.tags,
        "is_wiki": result.is_wiki,
        "support_count": result.support_count,
    }


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_optional_int(value: Any) -> int | None:
    text = _coerce_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"invalid integer: {value}") from exc


def _parse_positive_int(value: Any, *, default: int) -> int:
    parsed = _parse_optional_int(value)
    if parsed is None:
        return default
    if parsed <= 0:
        raise ValueError("integer value must be positive")
    return parsed


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = _coerce_text(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def _parse_choice(value: Any, *, default: str, choices: tuple[str, ...]) -> str:
    text = _coerce_text(value)
    if not text:
        return default
    if text not in choices:
        raise ValueError(f"invalid choice: {text}")
    return text


def _parse_scopes(value: Any) -> list[str]:
    return ["read"]


def _validate_community_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    allowed_hosts = {
        host.strip().lower()
        for host in os.environ.get("SHUIYUAN_ALLOWED_BASE_HOSTS", "shuiyuan.sjtu.edu.cn").split(",")
        if host.strip()
    }
    if parsed.scheme != "https" or not hostname or hostname not in allowed_hosts:
        allowed = ", ".join(sorted(allowed_hosts))
        raise ValueError(f"forbidden Shuiyuan base URL: {base_url} (allowed hosts: {allowed})")
