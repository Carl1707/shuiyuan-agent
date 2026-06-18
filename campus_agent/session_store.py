from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RECENT_TURN_LIMIT = 6


@dataclass(slots=True)
class SessionState:
    session_id: str
    title: str
    created_at: float
    updated_at: float
    session_summary: str
    current_topic: str
    active_entities: list[str]
    active_constraints: list[str]
    recent_turns: list[dict[str, Any]]
    artifacts: dict[str, Any]


class SessionMemoryStore:
    def __init__(self, path: str | Path = ".local/shuiyuan_agent.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_session(self, *, title: str = "新对话") -> SessionState:
        session_id = str(uuid.uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, title, created_at, updated_at, session_summary,
                    current_topic, active_entities, active_constraints
                ) VALUES (?, ?, ?, ?, '', '', '[]', '[]')
                """,
                (session_id, title.strip() or "新对话", now, now),
            )
        return self.get_session(session_id)

    def ensure_session(self, session_id: str | None) -> SessionState:
        session_id = (session_id or "").strip()
        if session_id:
            existing = self.get_session(session_id, missing_ok=True)
            if existing is not None:
                return existing
        return self.create_session()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, current_topic
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "current_topic": row["current_topic"],
            }
            for row in rows
        ]

    def get_session(self, session_id: str, *, missing_ok: bool = False) -> SessionState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                if missing_ok:
                    return None
                raise ValueError("unknown session_id")
            turns = conn.execute(
                """
                SELECT * FROM turns
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, _RECENT_TURN_LIMIT),
            ).fetchall()
            artifact_row = conn.execute(
                "SELECT * FROM session_artifacts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        recent_turns = [_turn_to_dict(turn) for turn in reversed(turns)]
        artifacts = _artifact_to_dict(artifact_row) if artifact_row is not None else {}
        return SessionState(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            session_summary=row["session_summary"],
            current_topic=row["current_topic"],
            active_entities=_json_list(row["active_entities"]),
            active_constraints=_json_list(row["active_constraints"]),
            recent_turns=recent_turns,
            artifacts=artifacts,
        )

    def record_turn(
        self,
        *,
        session_id: str,
        user_question: str,
        resolved_question: str,
        answer: str,
        answer_summary: str,
        memory_update: dict[str, Any],
        output: dict[str, Any],
    ) -> SessionState:
        now = time.time()
        state = self.get_session(session_id)
        title = state.title
        if title == "新对话":
            title = _make_title(user_question)
        active_entities = _string_list(memory_update.get("active_entities")) or state.active_entities
        active_constraints = _string_list(memory_update.get("active_constraints")) or state.active_constraints
        current_topic = str(memory_update.get("current_topic") or state.current_topic or "").strip()
        session_summary = str(memory_update.get("session_summary") or state.session_summary or "").strip()
        turn_id = str(uuid.uuid4())
        top_topic_urls = _top_topic_urls(output)
        supported_facts = _supported_facts(output)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, user_question, resolved_question, answer_summary,
                    answer, question_understanding, active_entities, active_constraints,
                    executed_queries, top_topic_urls, supported_facts, open_questions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    user_question,
                    resolved_question,
                    answer_summary,
                    answer,
                    _json(output.get("question_understanding", {})),
                    _json(active_entities),
                    _json(active_constraints),
                    _json(output.get("queries", [])),
                    _json(top_topic_urls),
                    _json(supported_facts),
                    _json(_string_list(memory_update.get("open_questions"))),
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?, session_summary = ?, current_topic = ?,
                    active_entities = ?, active_constraints = ?
                WHERE session_id = ?
                """,
                (
                    title,
                    now,
                    session_summary,
                    current_topic,
                    _json(active_entities),
                    _json(active_constraints),
                    session_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO session_artifacts (
                    session_id, last_search_plan, last_evidence_ledger,
                    last_expanded_topics, last_entity_set, last_entity_type, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_search_plan = excluded.last_search_plan,
                    last_evidence_ledger = excluded.last_evidence_ledger,
                    last_expanded_topics = excluded.last_expanded_topics,
                    last_entity_set = excluded.last_entity_set,
                    last_entity_type = excluded.last_entity_type,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    _json(output.get("search_plan", {})),
                    _json(output.get("evidence_ledger", {})),
                    _json(output.get("expanded_topics", [])),
                    _json(output.get("entity_set", [])),
                    str(output.get("entity_type", "") or ""),
                    now,
                ),
            )
        self._compress_old_turns(session_id)
        return self.get_session(session_id)

    def reset_session(self, session_id: str) -> SessionState:
        now = time.time()
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_artifacts WHERE session_id = ?", (session_id,))
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, session_summary = '', current_topic = '',
                    active_entities = '[]', active_constraints = '[]'
                WHERE session_id = ?
                """,
                (now, session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def _compress_old_turns(self, session_id: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT answer_summary, supported_facts
                FROM turns
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
                """,
                (session_id, _RECENT_TURN_LIMIT),
            ).fetchall()
            if not rows:
                return
            parts: list[str] = []
            for row in rows[:12]:
                summary = str(row["answer_summary"] or "").strip()
                facts = _json_list(row["supported_facts"])
                if summary:
                    parts.append(summary)
                parts.extend(str(item) for item in facts[:3])
            compact = "；".join(dict.fromkeys(part for part in parts if part))[:2000]
            conn.execute(
                "UPDATE sessions SET session_summary = ? WHERE session_id = ?",
                (compact, session_id),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    session_summary TEXT NOT NULL DEFAULT '',
                    current_topic TEXT NOT NULL DEFAULT '',
                    active_entities TEXT NOT NULL DEFAULT '[]',
                    active_constraints TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    user_question TEXT NOT NULL,
                    resolved_question TEXT NOT NULL,
                    answer_summary TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    question_understanding TEXT NOT NULL DEFAULT '{}',
                    active_entities TEXT NOT NULL DEFAULT '[]',
                    active_constraints TEXT NOT NULL DEFAULT '[]',
                    executed_queries TEXT NOT NULL DEFAULT '[]',
                    top_topic_urls TEXT NOT NULL DEFAULT '[]',
                    supported_facts TEXT NOT NULL DEFAULT '[]',
                    open_questions TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_artifacts (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
                    last_search_plan TEXT NOT NULL DEFAULT '{}',
                    last_evidence_ledger TEXT NOT NULL DEFAULT '{}',
                    last_expanded_topics TEXT NOT NULL DEFAULT '[]',
                    last_entity_set TEXT NOT NULL DEFAULT '[]',
                    last_entity_type TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                """
            )
            self._ensure_column(conn, "session_artifacts", "last_entity_set", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "session_artifacts", "last_entity_type", "TEXT NOT NULL DEFAULT ''")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def session_to_response(state: SessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "title": state.title,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "session_summary": state.session_summary,
        "current_topic": state.current_topic,
        "active_entities": state.active_entities,
        "active_constraints": state.active_constraints,
        "recent_turns": state.recent_turns,
        "artifacts": state.artifacts,
    }


def _turn_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "turn_id": row["turn_id"],
        "user_question": row["user_question"],
        "resolved_question": row["resolved_question"],
        "answer_summary": row["answer_summary"],
        "answer": row["answer"],
        "question_understanding": _json_obj(row["question_understanding"]),
        "active_entities": _json_list(row["active_entities"]),
        "active_constraints": _json_list(row["active_constraints"]),
        "executed_queries": _json_list(row["executed_queries"]),
        "top_topic_urls": _json_list(row["top_topic_urls"]),
        "supported_facts": _json_list(row["supported_facts"]),
        "open_questions": _json_list(row["open_questions"]),
        "created_at": row["created_at"],
    }


def _artifact_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "last_search_plan": _json_obj(row["last_search_plan"]),
        "last_evidence_ledger": _json_obj(row["last_evidence_ledger"]),
        "last_expanded_topics": _json_list(row["last_expanded_topics"]),
        "last_entity_set": _json_list(row["last_entity_set"]) if "last_entity_set" in row.keys() else [],
        "last_entity_type": str(row["last_entity_type"] or "") if "last_entity_type" in row.keys() else "",
        "updated_at": row["updated_at"],
    }


def _make_title(question: str) -> str:
    title = " ".join(question.split()).strip()
    return title[:28] or "新对话"


def _top_topic_urls(output: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for item in output.get("community_results", [])[:8]:
        url = getattr(item, "url", "")
        if url:
            urls.append(str(url))
    for item in output.get("results", [])[:8]:
        chunk = getattr(item, "chunk", None)
        metadata = getattr(chunk, "metadata", {}) if chunk is not None else {}
        url = metadata.get("post_url") or metadata.get("topic_url") or metadata.get("document_uri")
        if url:
            urls.append(str(url))
    return list(dict.fromkeys(urls))[:10]


def _supported_facts(output: dict[str, Any]) -> list[str]:
    ledger = output.get("evidence_ledger", {})
    if not isinstance(ledger, dict):
        return []
    facts: list[str] = []
    for key in ("current_answers", "direct_answers", "stable_support", "useful_support"):
        for item in ledger.get(key, []) if isinstance(ledger.get(key), list) else []:
            if isinstance(item, dict):
                claim = str(item.get("claim", "")).strip()
                if claim:
                    facts.append(claim)
            elif item:
                facts.append(str(item))
    return list(dict.fromkeys(facts))[:12]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_obj(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return list(dict.fromkeys(result))[:12]
