from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from campus_agent.tools import CommunitySearchResult


class LLMError(RuntimeError):
    """Raised when the configured LLM backend cannot be used."""


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    api_base: str = "https://models.sjtu.edu.cn/api/v1"
    model: str = "deepseek-chat"
    timeout_seconds: int = 60


@dataclass(slots=True)
class ShuiyuanSearchPlan:
    intent: str
    bridges: list[str]
    queries: list[str]
    question_understanding: dict[str, object] = field(default_factory=dict)
    search_views: list[str] = field(default_factory=list)
    coverage_axes: list[str] = field(default_factory=list)
    answer_contract: dict[str, object] = field(default_factory=dict)
    query_details: list[dict[str, str]] = field(default_factory=list)
    selected_query_details: list[dict[str, str]] = field(default_factory=list)
    rejected_query_details: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceAssessment:
    can_answer: bool
    covered_requirements: list[str]
    missing_requirements: list[str]
    next_queries: list[dict[str, str]]
    topics_to_expand: list[dict[str, str]]
    question_shape: str = "single_fact"
    topic_scores: list[dict[str, object]] = field(default_factory=list)
    summary: str = ""
    evidence_roles: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ContextualQuestion:
    is_followup: bool
    resolved_question: str
    current_topic: str = ""
    turn_operation: str = "new_topic"
    reuse_strategy: str = "full_refresh"
    target_entity_type: str = "unknown"
    required_attributes: list[str] = field(default_factory=list)
    candidate_filters: list[str] = field(default_factory=list)
    active_entities: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    dropped_context: list[str] = field(default_factory=list)
    retrieval_instruction: str = ""
    answer_style_instruction: str = ""
    session_summary: str = ""
    open_questions: list[str] = field(default_factory=list)


_DEFAULT_QUESTION_UNDERSTANDING = {
    "user_goal": "",
    "result_shape": "single_best",
    "search_intent": "找最直接可用的校园社区证据",
    "coverage_expectation": "narrow",
    "organization_strategy": "flat_summary",
    "freshness_sensitivity": "medium",
    "evidence_priority": [],
    "known_risks": [],
}


def resolve_contextual_question(
    *,
    question: str,
    conversation_context: dict[str, object] | None = None,
    session_system_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> ContextualQuestion:
    """Resolve short follow-up questions against session-local memory."""

    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    context = conversation_context or {}
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户当前问题：{question}\n"
        f"全局对话偏好：{session_system_prompt or ''}\n"
        f"会话上下文：{json.dumps(context, ensure_ascii=False)}\n\n"
        "请判断当前问题是否是在追问本会话中的已有主题。"
        "如果是追问，请把省略的对象、校区、时间、主题和约束补全为一个可独立检索的问题。"
        "如果用户切换到了新话题，不要继承旧主题约束。"
        "全局对话偏好只能作为回答风格或来源偏好的提示，不能覆盖证据规则。"
        "输出 JSON，字段固定为 is_followup、resolved_question、current_topic、turn_operation、"
        "reuse_strategy、target_entity_type、required_attributes、candidate_filters、"
        "active_entities、active_constraints、dropped_context、retrieval_instruction、"
        "answer_style_instruction、session_summary、open_questions。"
        "resolved_question 必须是自然中文问题，适合继续交给 Shuiyuan 搜索规划器。"
        "active_entities 和 active_constraints 只保留当前轮仍然有效的信息。"
        "turn_operation 只能是 new_topic、refine_scope、filter_candidates、compare_candidates、fill_missing_attributes。"
        "reuse_strategy 只能是 reuse_only、reuse_then_expand、full_refresh。"
        "target_entity_type 只能是 person、lab、course、location、department、unknown。"
        "如果当前问题是在上一轮候选对象里继续筛选、比较或补属性，优先使用 reuse_only 或 reuse_then_expand，不要默认 full_refresh。"
    )
    parsed = _request_json_object(
        config,
        system="你是对话上下文解析器。严格只输出 JSON 对象。",
        prompt=prompt,
        temperature=0.1,
    )
    return _normalize_contextual_question(parsed, question)


def synthesize_entities_from_evidence(
    *,
    question: str,
    question_understanding: dict[str, object] | None,
    contextual_question: ContextualQuestion,
    evidence_items: list[dict[str, object]],
    structured_evidence: dict[str, object] | None = None,
    previous_entity_set: list[dict[str, object]] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户问题：{question}\n"
        f"问题理解：{json.dumps(question_understanding or {}, ensure_ascii=False)}\n"
        f"多轮操作：{json.dumps(_contextual_question_to_dict(contextual_question), ensure_ascii=False)}\n"
        f"上一轮对象集合：{json.dumps(previous_entity_set or [], ensure_ascii=False)}\n"
        f"结构化正文证据：{json.dumps(structured_evidence or {}, ensure_ascii=False)}\n"
        f"候选证据：{json.dumps(evidence_items[:40], ensure_ascii=False)}\n\n"
        "请把证据整理成对象集合，而不是帖子集合。"
        "如果这是追问筛选或补属性，优先复用上一轮对象集合，只在当前证据直接支持时更新或过滤。"
        "输出 JSON，字段固定为 entity_type、entity_set、entity_merge_notes、missing_attributes、insufficient_entities。"
        "entity_set 每项包含 entity_id、entity_name、entity_type、aliases、attributes、evidence_urls、evidence_ids、confidence。"
        "attributes 是对象属性字典，例如学院、方向、岗位、窗口、时间、地点等。"
        "如果证据只说明某对象与主题相关，可以保留，但不要把整段帖子摘要塞进 attributes。"
    )
    parsed = _request_json_object(
        config,
        system="你是对象级证据归并器。严格只输出 JSON 对象。",
        prompt=prompt,
        temperature=0.1,
    )
    return _normalize_entity_synthesis(parsed, contextual_question)


def load_local_env() -> None:
    for env_name in (".env.local", ".env"):
        path = Path(env_name)
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_llm_config(
    model: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> LLMConfig:
    load_local_env()
    api_key = (api_key if api_key is not None else os.environ.get("SJTU_LLM_API_KEY", "")).strip()
    if not api_key:
        raise LLMError("missing SJTU_LLM_API_KEY")
    api_base = (
        api_base
        if api_base is not None
        else os.environ.get("SJTU_LLM_API_BASE", "https://models.sjtu.edu.cn/api/v1")
    ).strip().rstrip("/")
    _validate_llm_api_base(api_base)
    resolved_model = (model or os.environ.get("SJTU_LLM_MODEL") or "deepseek-chat").strip()
    timeout_seconds = timeout_seconds if timeout_seconds is not None else int(
        os.environ.get("SJTU_LLM_TIMEOUT_SECONDS", "60")
    )
    return LLMConfig(
        api_key=api_key,
        api_base=api_base,
        model=resolved_model,
        timeout_seconds=timeout_seconds,
    )


def _validate_llm_api_base(api_base: str) -> None:
    parsed = urlparse(api_base)
    hostname = (parsed.hostname or "").lower()
    allowed_hosts = {
        host.strip().lower()
        for host in os.environ.get("SJTU_LLM_ALLOWED_BASE_HOSTS", "models.sjtu.edu.cn").split(",")
        if host.strip()
    }
    if parsed.scheme != "https" or not hostname or hostname not in allowed_hosts:
        allowed = ", ".join(sorted(allowed_hosts))
        raise LLMError(f"forbidden llm api base: {api_base} (allowed hosts: {allowed})")


def generate_shuiyuan_search_plan(
    *,
    question: str,
    question_type: str = "",
    max_queries: int = 15,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> ShuiyuanSearchPlan:
    """Use the LLM to plan bridged Shuiyuan search queries."""

    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    current_date = date.today().isoformat()
    current_year = date.today().year
    prompt = "\n".join(
        [
            f"当前日期：{current_date}。当前年份是 {current_year}。",
            f"用户原问题：{question}",
            f"问题类型提示：{question_type or '未提供，请自行判断'}",
            "请把这个问题理解成一个校园社区搜索任务，而不是直接回答任务。",
            "你需要推断这个问题在校园语境下可能涉及的责任部门、地点、动作、流程词和社区口语表达。",
            (
                "如果问题涉及具体人物，请让 LLM 推断社区里可能使用的人名变体，例如中文全名、"
                "拼音或英文首字母、用户名式简称，以及有充分语境依据的外号；"
                "把不同变体拆成短查询，但不要凭空编造外号。"
            ),
            "请先动态定义什么信息才算真正回答了用户问题，不要套用预设问题类型。",
            (
                "请输出 JSON 对象，字段固定为 intent、question_understanding、search_views、coverage_axes、"
                "bridges、answer_contract、queries、selected_queries、rejected_queries。"
                f"queries 最多 {max_queries} 条。"
            ),
            (
                "question_understanding 包含 user_goal、result_shape、search_intent、coverage_expectation、"
                "organization_strategy、freshness_sensitivity、evidence_priority、known_risks。"
                "result_shape 只能是 single_best、actionable_steps、broad_options、current_rule、compare_options。"
                "coverage_expectation 只能是 narrow、medium、wide。"
                "organization_strategy 只能是 flat_summary、by_steps、by_entities、by_current_vs_history、by_options。"
                "freshness_sensitivity 只能是 low、medium、high。"
            ),
            (
                "answer_contract 包含 user_need、required_facts、helpful_facts、insufficient_evidence、"
                "requires_latest_evidence、hard_constraints；这些内容必须针对当前问题动态生成。"
            ),
            (
                "如果用户看起来是在问‘有没有、有哪些、哪些、谁在做、哪里可以、求推荐、推荐一下’等宽范围发现问题，"
                "请优先理解为需要 broad_options，而不是 single_best；"
                "此时 organization_strategy 优先用 by_entities 或 by_options，coverage_expectation 优先用 wide。"
            ),
            (
                "如果问题涉及政策、资格条件、申请流程、时间节点、门槛、费用、开放安排、"
                "是否还能办、什么时候办、是否有最新变化等，请视为强时效或规则敏感问题。"
            ),
            (
                "对强时效或规则敏感问题，搜索计划必须同时覆盖：当前周期事实、近期社区经验、"
                "宽泛核心表述和具体执行细节。不能让所有查询都变成通知、公示或正式部门查询。"
            ),
            (
                "对这类问题，不要把旧经验帖里的门槛、日期、费用、资格条件直接当成当前硬规则；"
                "应把‘是否需要最新通知或当年安排’写入 answer_contract。"
            ),
            (
                "queries 中每项包含 query、purpose、lane。lane 只能是 direct、current_cycle、"
                "community_experience、bridge。purpose 说明该查询准备验证或补充什么。"
            ),
            (
                "对于 broad_options 问题，请先定义 search_views 和 coverage_axes。"
                "search_views 是这题应该从哪些视角搜索，例如方向、对象、院系、体验、地点、流程。"
                "coverage_axes 是最终希望覆盖的维度，例如老师、实验室、院系、研究方向、地点、课程类别。"
                "queries 应分散覆盖这些不同视角，避免全部围绕同一个短词。"
            ),
            "queries 必须模拟水源用户发帖和搜索时的短口语表达，每条通常为 1 至 2 个短词组。",
            "水源接近子串关键词搜索：避免学校全称、长句、正式公文表达和过多关键词堆叠。",
            (
                f"除非用户明确询问历史年份，否则不要生成 {current_year} 年之前的年份查询；"
                f"强时效问题若使用年份，只能在 current_cycle 查询中使用当前年份 {current_year}。"
                "direct、community_experience、bridge 查询不要带年份，以免错过近期综合帖和经验帖。"
            ),
            "queries 必须多样化，覆盖不同检索目的，而不是简单改写同一句话。",
            (
                "优先保留宽泛核心 query 和社区经验 query，因为它们常能召回高质量综合帖；"
                "再用当前周期和桥接 query 补充时效与执行细节。"
            ),
            "selected_queries 从 queries 中选出 5 至 7 条最容易命中且信息增益不同的查询。",
            "rejected_queries 记录未选查询及原因，每项包含 query 和 reason。",
            "不要编造结论，只生成便于在 Shuiyuan 搜索框中检索的短中文查询。",
            (
                '只输出 JSON，例如：'
                '{"intent":"校园失物与应急求助",'
                '"question_understanding":{"user_goal":"找到可执行的求助方式","result_shape":"actionable_steps","search_intent":"定位负责部门和可执行动作","coverage_expectation":"medium","organization_strategy":"by_steps","freshness_sensitivity":"medium","evidence_priority":["负责部门","近期案例"],"known_risks":["不要承诺一定能找回"]},'
                '"search_views":["负责部门","类似案例"],'
                '"coverage_axes":["部门","动作"],'
                '"bridges":["保卫处","失物招领","打捞","求助"],'
                '"answer_contract":{"user_need":"找到可执行的求助方式",'
                '"required_facts":["可联系的人员或部门","可执行的处理步骤"],'
                '"helpful_facts":["类似案例"],"insufficient_evidence":["普通手机维修建议"],'
                '"requires_latest_evidence":false,"hard_constraints":["不能承诺一定能找回手机"]},'
                '"queries":[{"query":"手机 保卫处","purpose":"寻找负责部门","lane":"bridge"},'
                '{"query":"手机 掉湖里","purpose":"寻找类似案例","lane":"community_experience"}],'
                '"selected_queries":[{"query":"手机 保卫处","purpose":"寻找负责部门","lane":"bridge"}],'
                '"rejected_queries":[{"query":"上海交通大学手机落水处理","reason":"过长且过于正式"}]}'
            ),
        ]
    )
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是校园社区搜索规划器。"
                    "你的任务不是回答问题，而是先理解校园事务意图，再补出桥接概念，"
                    "最后生成多样化的 Shuiyuan 搜索 queries。"
                    "严格只输出 JSON 对象。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    body = _post_chat_completion(config, payload)
    try:
        raw = str(body["choices"][0]["message"]["content"]).strip()
        parsed = json.loads(_strip_json_fence(raw))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError(f"unexpected query planning response: {body}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("query planning response is not a JSON object")
    intent = " ".join(str(parsed.get("intent", "")).split()).strip()
    raw_bridges = parsed.get("bridges", [])
    raw_queries = parsed.get("queries", [])
    raw_selected = parsed.get("selected_queries", [])
    raw_rejected = parsed.get("rejected_queries", [])
    raw_contract = parsed.get("answer_contract", {})
    raw_understanding = parsed.get("question_understanding", {})
    raw_search_views = parsed.get("search_views", [])
    raw_coverage_axes = parsed.get("coverage_axes", [])
    if not isinstance(raw_bridges, list) or not isinstance(raw_queries, list):
        raise LLMError("query planning response must contain list fields")
    bridges = _dedupe_text_items(raw_bridges, limit=8)
    question_understanding = _normalize_question_understanding(raw_understanding, question)
    answer_contract = _normalize_answer_contract(raw_contract, question)
    query_details = _align_query_years(
        _normalize_query_details(raw_queries, limit=max_queries),
        question=question,
        answer_contract=answer_contract,
        current_year=current_year,
    )
    queries = [item["query"] for item in query_details]
    if not queries:
        raise LLMError("query planning returned no usable queries")
    return ShuiyuanSearchPlan(
        intent=intent or "校园事务搜索",
        bridges=bridges,
        queries=queries,
        question_understanding=question_understanding,
        search_views=_dedupe_text_items(_list_value(raw_search_views), limit=8),
        coverage_axes=_dedupe_text_items(_list_value(raw_coverage_axes), limit=8),
        answer_contract=answer_contract,
        query_details=query_details,
        selected_query_details=_balance_selected_queries(
            query_details,
            _align_query_years(
                _normalize_query_details(raw_selected, limit=7),
                question=question,
                answer_contract=answer_contract,
                current_year=current_year,
            ),
            answer_contract=answer_contract,
            limit=7,
        ),
        rejected_query_details=_normalize_rejected_queries(raw_rejected, limit=max_queries),
    )


def classify_candidate_objects(
    *,
    question: str,
    question_understanding: dict[str, object],
    community_results: list[CommunitySearchResult],
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> list[dict[str, object]]:
    if not community_results:
        return []
    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    evidence = [
        {
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "tags": item.tags,
            "is_wiki": item.is_wiki,
        }
        for item in community_results[:30]
    ]
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户问题：{question}\n"
        f"问题理解：{json.dumps(question_understanding, ensure_ascii=False)}\n"
        f"候选帖子：{json.dumps(evidence, ensure_ascii=False)}\n\n"
        "请从“当前问题需要找什么对象”的角度，对每条候选做对象归并与作用分析。"
        "输出 JSON，字段固定为 candidate_profiles。"
        "candidate_profiles 每项包含 url、primary_object、object_kind、scope、coverage_tags、redundant_with、reason。"
        "object_kind 只能是 person、lab、team、department、place、course、service、unknown。"
        "scope 只能是 comprehensive、specialized、single_case、discussion、unknown。"
        "coverage_tags 是与当前问题相关的对象标签、方向标签或院系标签。"
        "如果两条帖子明显围绕同一个对象，后者的 redundant_with 应填主对象名；否则留空。"
        "对于需要 broad_options 的问题，优先把综合帖标成 comprehensive，把单对象帖子标成 specialized 或 single_case。"
        "不要编造超出标题摘要的信息。严格只输出 JSON。"
    )
    parsed = _request_json_object(
        config,
        system="你是校园社区候选结果分析器。请按当前问题需要的对象范围，对标题摘要做对象归并和作用判断。严格输出 JSON。",
        prompt=prompt,
        temperature=0.1,
    )
    return _normalize_candidate_profiles(_list_value(parsed.get("candidate_profiles")), evidence)


def assess_shuiyuan_evidence(
    *,
    question: str,
    answer_contract: dict[str, object],
    question_understanding: dict[str, object] | None,
    community_results: list[CommunitySearchResult],
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> EvidenceAssessment:
    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    evidence = [
        {
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "tags": item.tags,
            "is_wiki": item.is_wiki,
        }
        for item in community_results[:30]
    ]
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户问题：{question}\n"
        f"问题理解：{json.dumps(question_understanding or {}, ensure_ascii=False)}\n"
        f"动态答案契约：{json.dumps(answer_contract, ensure_ascii=False)}\n"
        f"当前 Shuiyuan 搜索证据：{json.dumps(evidence, ensure_ascii=False)}\n\n"
        "先判断这个问题的形状，再判断每条证据在回答当前问题时的角色，而不只是主题相关。"
        "如果 answer_contract 表示这是强时效或规则敏感问题，"
        "则旧年份经验、未标明年份的门槛、个人推测不能直接算作 direct_answer。"
        "对于本校政策、流程、资格或校园服务问题，必须先判断证据的适用主体；"
        "其他学校、其他机构的政策即使更新、更具体，也只能算 background_only 或 irrelevant，"
        "不能作为 direct_answer，也不应优先展开。"
        "输出 JSON，字段为 question_shape、can_answer、covered_requirements、missing_requirements、"
        "topic_scores、evidence_roles、summary。"
        "question_shape 只能是 single_fact、procedure、enumeration、comparison、recommendation、time_sensitive_rule。"
        "evidence_roles 每项包含 url、role、reason；role 只能是 direct_answer、"
        "useful_support、background_only、irrelevant。"
        "direct_answer 必须直接满足用户的正向需求；只说明哪里不能去属于 useful_support。"
        "对强时效或规则敏感问题，若证据只支持‘往年经验’或‘可能如此’，应降为 useful_support 或 background_only。"
        "但当前周期的社区经验、近期实践讨论和高质量综合帖可以成为 direct_answer，"
        "不能因为它不是通知或官方转帖就自动降级。"
        "topic_scores 只对最值得展开的帖子评分，每项包含 url、expand_score、reason、expected_value。"
        "expand_score 范围 0 到 1。expected_value 是对象，包含 entity_discovery、rule_detail、actionability、freshness_signal、complementarity，"
        "每个字段只能是 high、medium、low。"
        "对 enumeration / comparison / recommendation 问题，要优先选择能补充不同实体或不同选项的帖子，而不是重复同一地点。"
        "如果用户问的是宽泛或全校层面问题，优先展开综合经验帖、汇总帖和覆盖多个对象的讨论；"
        "单一学院、单一部门或单一地点的帖子只能作为具体例子，不能压过范围更匹配的综合帖。"
    )
    parsed = _request_json_object(
        config,
        system=(
            "你是社区搜索结果分类器。区分直接答案、辅助信息、背景和无关内容，"
            "并评估每个帖子是否值得展开阅读正文。严格只输出 JSON。"
        ),
        prompt=prompt,
        temperature=0.1,
    )
    question_shape = _normalize_question_shape(parsed.get("question_shape"))
    topic_scores = _normalize_topic_scores(_list_value(parsed.get("topic_scores")), evidence)
    return EvidenceAssessment(
        question_shape=question_shape,
        can_answer=bool(parsed.get("can_answer")),
        covered_requirements=_dedupe_text_items(_list_value(parsed.get("covered_requirements")), limit=12),
        missing_requirements=_dedupe_text_items(_list_value(parsed.get("missing_requirements")), limit=12),
        next_queries=[],
        topics_to_expand=[
            {
                "url": str(item.get("url", "")),
                "reason": str(item.get("reason", "")),
                "expand_score": float(item.get("expand_score", 0.0)),
                "expected_value": item.get("expected_value", {}),
            }
            for item in topic_scores
        ],
        topic_scores=topic_scores,
        summary=" ".join(str(parsed.get("summary", "")).split()).strip(),
        evidence_roles=_normalize_evidence_roles(_list_value(parsed.get("evidence_roles")), evidence),
    )


def extract_structured_community_evidence(
    *,
    question: str,
    question_shape: str,
    answer_contract: dict[str, object],
    question_understanding: dict[str, object] | None,
    expanded_topics: list[dict[str, object]],
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    if not expanded_topics:
        return {
            "entities": [],
            "actions": [],
            "rules": [],
            "time_constraints": [],
            "uncertainties": [],
            "current_cycle_evidence": [],
            "stable_practice_evidence": [],
            "historical_or_uncertain_evidence": [],
        }
    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户问题：{question}\n"
        f"问题形状：{question_shape}\n"
        f"问题理解：{json.dumps(question_understanding or {}, ensure_ascii=False)}\n"
        f"动态答案契约：{json.dumps(answer_contract, ensure_ascii=False)}\n"
        f"已展开帖子正文：{json.dumps(expanded_topics[:8], ensure_ascii=False)}\n\n"
        "请把正文中的可用证据抽成结构化槽位。"
        "输出 JSON，字段固定为 entities、actions、rules、time_constraints、uncertainties、"
        "current_cycle_evidence、stable_practice_evidence、historical_or_uncertain_evidence。"
        "entities 每项包含 name、kind、evidence_url、facts、freshness_level、evidence_time。"
        "actions 每项包含 action、evidence_url、details、freshness_level、evidence_time。"
        "rules 每项包含 rule、evidence_url、details、freshness_level、evidence_time。"
        "time_constraints 每项包含 constraint、evidence_url、details、freshness_level、evidence_time。"
        "uncertainties 每项包含 uncertainty、evidence_url、details、freshness_level、evidence_time。"
        "freshness_level 只能是 current_cycle、stable_practice、historical_or_uncertain。"
        "current_cycle_evidence、stable_practice_evidence、historical_or_uncertain_evidence "
        "每项包含 claim、evidence_url、details。"
        "若问题是 enumeration，优先抽取不同地点/对象；若问题是 procedure，优先抽取步骤和限制。"
        "若问题是 time_sensitive_rule，优先区分当前规则、社区准备经验和旧经验。"
        "如果问题范围宽泛，而正文中同时有综合经验帖和单一学院/部门/地点的专门帖，"
        "应先从综合帖跨回复提炼共性、差异和多种案例，再把专门帖内容作为例子；"
        "不要因为专门帖单条回复更具体，就让它代表全校或所有对象。"
        "不要编造没有证据支持的字段，字段值必须能在给定正文中找到依据。"
    )
    parsed = _request_json_object(
        config,
        system="你是社区正文结构化抽取器。只依据给定帖子正文抽取槽位，严格输出 JSON。",
        prompt=prompt,
        temperature=0.1,
    )
    return {
        "entities": _normalize_structured_group(_list_value(parsed.get("entities")), kind="entity"),
        "actions": _normalize_structured_group(_list_value(parsed.get("actions")), kind="action"),
        "rules": _normalize_structured_group(_list_value(parsed.get("rules")), kind="rule"),
        "time_constraints": _normalize_structured_group(_list_value(parsed.get("time_constraints")), kind="time"),
        "uncertainties": _normalize_structured_group(_list_value(parsed.get("uncertainties")), kind="uncertainty"),
        "current_cycle_evidence": _normalize_structured_summary_group(
            _list_value(parsed.get("current_cycle_evidence"))
        ),
        "stable_practice_evidence": _normalize_structured_summary_group(
            _list_value(parsed.get("stable_practice_evidence"))
        ),
        "historical_or_uncertain_evidence": _normalize_structured_summary_group(
            _list_value(parsed.get("historical_or_uncertain_evidence"))
        ),
    }


def build_evidence_ledger(
    *,
    question: str,
    question_shape: str,
    answer_contract: dict[str, object],
    question_understanding: dict[str, object] | None,
    evidence_items: list[dict[str, object]],
    structured_evidence: dict[str, object] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户问题：{question}\n"
        f"问题形状：{question_shape}\n"
        f"问题理解：{json.dumps(question_understanding or {}, ensure_ascii=False)}\n"
        f"动态答案契约：{json.dumps(answer_contract, ensure_ascii=False)}\n"
        f"结构化正文证据：{json.dumps(structured_evidence or {}, ensure_ascii=False)}\n"
        f"候选证据：{json.dumps(evidence_items[:40], ensure_ascii=False)}\n\n"
        "对摘要和回复做最终证据角色分类并建立事实账本。"
        "如果这是强时效或规则敏感问题，禁止把旧帖子中的固定门槛、日期、费用、资格条件"
        "直接整理为当前确定事实，除非证据明确显示这是当年/最新规则。"
        "输出 JSON，字段为 current_answers、stable_support、historical_background、"
        "direct_answers、useful_support、background_only、remaining_unknowns。"
        "current_answers、stable_support、historical_background、direct_answers 和 useful_support 每项包含 claim、"
        "evidence_ids、confidence。只说明不可用地点、关闭时间或限制的信息必须放入 "
        "useful_support，不能放入 direct_answers。"
        "current_answers 用于当前周期内更可能仍然成立的内容；"
        "stable_support 用于长期稳定的社区经验或准备建议；"
        "historical_background 用于可能过期、年份冲突或只能做背景参考的内容。"
        "对于 enumeration 问题，允许 direct_answers 由多个实体条目组成；如果结构化正文证据已经抽出了不同对象，不要因为单条证据不完整而全部降为 useful_support。"
        "对于宽泛问题，综合帖中跨对象、跨回复重复出现的共性应优先进入 direct_answers；"
        "单一学院、部门、地点或个案的细节只能作为例子或 useful_support，除非用户明确询问该对象。"
        "对于 time_sensitive_rule 问题，和现行规则相关的固定门槛、流程、时间必须优先进入 current_answers；"
        "准备建议、常见误区和经验补充优先进入 stable_support；过往门槛或年份冲突内容进入 historical_background。"
    )
    parsed = _request_json_object(
        config,
        system="你是严格的证据整理器。每个事实都必须由给定证据直接支持。严格只输出 JSON。",
        prompt=prompt,
        temperature=0.1,
    )
    return {
        "current_answers": _list_value(parsed.get("current_answers"))[:10],
        "stable_support": _list_value(parsed.get("stable_support"))[:10],
        "historical_background": _list_value(parsed.get("historical_background"))[:10],
        "direct_answers": _list_value(parsed.get("direct_answers"))[:10],
        "useful_support": _list_value(parsed.get("useful_support"))[:10],
        "background_only": _list_value(parsed.get("background_only"))[:10],
        "remaining_unknowns": _dedupe_text_items(_list_value(parsed.get("remaining_unknowns")), limit=10),
    }


def generate_verified_answer(
    *,
    question: str,
    question_shape: str,
    answer_contract: dict[str, object],
    question_understanding: dict[str, object] | None,
    evidence_ledger: dict[str, object],
    evidence_items: list[dict[str, object]],
    structured_evidence: dict[str, object] | None = None,
    entity_set: list[dict[str, object]] | None = None,
    turn_operation: str = "new_topic",
    required_attributes: list[str] | None = None,
    session_answer_instruction: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    config = load_llm_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    prompt = (
        f"当前日期：{date.today().isoformat()}。\n"
        f"用户问题：{question}\n"
        f"问题形状：{question_shape}\n"
        f"问题理解：{json.dumps(question_understanding or {}, ensure_ascii=False)}\n"
        f"动态答案契约：{json.dumps(answer_contract, ensure_ascii=False)}\n"
        f"当前对象集合：{json.dumps(entity_set or [], ensure_ascii=False)}\n"
        f"当前追问操作：{turn_operation}\n"
        f"本轮重点补全属性：{json.dumps(required_attributes or [], ensure_ascii=False)}\n"
        f"结构化正文证据：{json.dumps(structured_evidence or {}, ensure_ascii=False)}\n"
        f"事实账本：{json.dumps(evidence_ledger, ensure_ascii=False)}\n"
        f"证据索引：{json.dumps(evidence_items[:40], ensure_ascii=False)}\n\n"
        f"全局对话偏好：{session_answer_instruction or ''}\n"
        "请生成自然、直接、可执行的中文回答。优先使用 current_answers 和 direct_answers 正面回答用户；"
        "如果问题形状是 enumeration，可把 stable_support 和 useful_support 中能够补全实体信息的内容合并进主体答案，"
        "但当前预约规则、开放时间、人数限制等若无法确认，要单独标出“当前未完全确认”；"
        "如果问题理解中的 organization_strategy 是 by_entities 或 by_options，优先按对象或选项分条组织；"
        "不要按帖子顺序拼接内容。"
        "如果当前对象集合非空，且 turn_operation 不是 new_topic，优先把回答组织成“对象级总结”，"
        "也就是先按对象名称列出，再只补该对象当前能确认的属性；"
        "不要把帖子摘要原样抄进答案，也不要把对象集合退化回帖子列表。"
        "如果 required_attributes 非空，优先回答这些属性，无法确认的属性应明确写“当前帖子未确认”。"
        "对于 refine_scope、filter_candidates、compare_candidates、fill_missing_attributes 这几类追问，"
        "默认继承上一轮对象范围，除非新证据明确引入了新的关键对象。"
        "如果问题形状是 time_sensitive_rule，先写“当前能确认的规则/流程”，再写“社区经验补充/准备建议”，"
        "historical_background 只能用于解释“过去有人这样说过，但不能视为现行规则”；"
        "time_sensitive_rule 回答应先用一两句话说明当前状态和共通流程，并明确所有日期与规则对应的证据年份；"
        "若不同学院、部门、场地或具体对象的要求不同，要显式提醒用户分别核对，不能暗示存在统一门槛；"
        "如果用户问题范围宽泛，先回答综合帖支持的共性结论，再列举专门帖中的具体案例；"
        "不得用单一学院、部门、地点或个案替代整体答案。"
        "除非用户要求完整政策清单，否则不要逐条复述所有例外和限制，优先保留对用户决策有帮助的信息；"
        "近期社区经验可以作为可操作建议，但必须与当前规则分段表达。"
        "useful_support 其余情况下只能作为补充、限制或排除项，不能伪装成主答案；"
        "background_only 不得进入答案。若没有 direct_answers，明确说明没有找到直接答案。"
        "没有证据的内容必须删除，不要机械展示证据编号。"
        "如果没有当前证据，不得推断某项工作尚未启动、通知尚未发布或通常会在某月进行。"
        "如果 answer_contract.requires_latest_evidence 为 true，"
        "则禁止把旧经验、未注明年份的门槛、论坛猜测写成当前硬规则。"
        "除非证据直接支持，否则不要输出固定的学积分/GPA门槛、明确日期、固定费用或资格条件。"
        "若证据中只有往年经验或年份冲突，必须明确写出“以当年/最新通知为准”或“现有帖子不足以确认当前规则”。"
        "answer_contract.hard_constraints 中列出的限制必须遵守。"
        "全局对话偏好只能影响表达风格、详略和组织方式，不能覆盖证据约束。"
        "回答结尾单独列出“参考帖子”，使用证据索引中的标题与可跳转 URL。"
        "如果对象集合里已经有 evidence_urls，可优先从这些 URL 里选参考帖子。"
    )
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "你是上海交通大学校园事务问答助手，只能使用事实账本中的已支持事实回答。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "stream": False,
    }
    body = _post_chat_completion(config, payload)
    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected verified answer response: {body}") from exc


def _request_json_object(
    config: LLMConfig,
    *,
    system: str,
    prompt: str,
    temperature: float,
) -> dict[str, object]:
    body = _post_chat_completion(
        config,
        {
            "model": config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        },
    )
    try:
        parsed = json.loads(_strip_json_fence(str(body["choices"][0]["message"]["content"]).strip()))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError(f"unexpected structured llm response: {body}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("structured llm response is not a JSON object")
    return parsed


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_query_details(items: list[object], *, limit: int) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            query = " ".join(str(item.get("query", "")).split()).strip()
            purpose = " ".join(str(item.get("purpose", "")).split()).strip()
            lane = _normalize_query_lane(item.get("lane"), query=query, purpose=purpose)
        else:
            query = " ".join(str(item).split()).strip()
            purpose = ""
            lane = "direct"
        query = _compact_shuiyuan_query(query, lane=lane)
        if not query or query in seen:
            continue
        seen.add(query)
        details.append({"query": query, "purpose": purpose, "lane": lane})
        if len(details) >= limit:
            break
    return details


def _normalize_query_lane(value: object, *, query: str, purpose: str) -> str:
    lane = str(value or "").strip()
    allowed = {"direct", "current_cycle", "community_experience", "bridge"}
    normalized = f"{query} {purpose}"
    if any(term in normalized for term in ("经验", "心得", "避坑", "分享", "求助", "建议", "讨论")):
        return "community_experience"
    return lane if lane in allowed else "direct"


def _compact_shuiyuan_query(query: str, *, lane: str) -> str:
    parts = query.split()
    if len(parts) <= 2:
        return query
    if lane == "community_experience":
        return f"{parts[0]} {parts[-1]}"
    return " ".join(parts[:2])


def _align_query_years(
    items: list[dict[str, str]],
    *,
    question: str,
    answer_contract: dict[str, object],
    current_year: int,
) -> list[dict[str, str]]:
    if not bool(answer_contract.get("requires_latest_evidence")):
        return items
    years_in_question = {int(value) for value in re.findall(r"\b20\d{2}\b", question)}
    aligned: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        query = item["query"]
        if not years_in_question:
            if item.get("lane") == "current_cycle":
                query = re.sub(r"\b20\d{2}\b", str(current_year), query)
            else:
                query = re.sub(r"\b20\d{2}\b", "", query)
                query = " ".join(query.split())
        if query in seen:
            continue
        seen.add(query)
        aligned.append({**item, "query": query})
    return aligned


def _balance_selected_queries(
    all_queries: list[dict[str, str]],
    selected_queries: list[dict[str, str]],
    *,
    answer_contract: dict[str, object],
    limit: int,
) -> list[dict[str, str]]:
    selected = list(selected_queries or all_queries[:limit])
    if bool(answer_contract.get("requires_latest_evidence")):
        present_lanes = {item.get("lane", "direct") for item in selected}
        for required_lane in ("direct", "community_experience", "current_cycle"):
            if required_lane in present_lanes:
                continue
            candidate = next((item for item in all_queries if item.get("lane") == required_lane), None)
            if candidate is not None:
                selected.append(candidate)
                present_lanes.add(required_lane)
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in selected:
        if item["query"] in seen:
            continue
        seen.add(item["query"])
        deduped.append(item)
    return deduped[:limit]


def _normalize_rejected_queries(items: object, *, limit: int) -> list[dict[str, str]]:
    rejected: list[dict[str, str]] = []
    for item in _list_value(items):
        if not isinstance(item, dict):
            continue
        query = " ".join(str(item.get("query", "")).split()).strip()
        reason = " ".join(str(item.get("reason", "")).split()).strip()
        if query:
            rejected.append({"query": query, "reason": reason})
        if len(rejected) >= limit:
            break
    return rejected


def _normalize_evidence_roles(
    items: list[object],
    evidence: list[dict[str, object]],
) -> list[dict[str, str]]:
    allowed_urls = {str(item.get("url", "")) for item in evidence}
    allowed_roles = {"direct_answer", "useful_support", "background_only", "irrelevant"}
    roles: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        role = str(item.get("role", "")).strip()
        if not url or url not in allowed_urls or role not in allowed_roles or url in seen:
            continue
        seen.add(url)
        roles.append(
            {
                "url": url,
                "role": role,
                "reason": " ".join(str(item.get("reason", "")).split()).strip(),
            }
        )
    return roles


def _normalize_answer_contract(value: object, question: str) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    return {
        "user_need": " ".join(str(raw.get("user_need") or question).split()).strip(),
        "required_facts": _dedupe_text_items(_list_value(raw.get("required_facts")), limit=8),
        "helpful_facts": _dedupe_text_items(_list_value(raw.get("helpful_facts")), limit=8),
        "insufficient_evidence": _dedupe_text_items(_list_value(raw.get("insufficient_evidence")), limit=8),
        "requires_latest_evidence": bool(raw.get("requires_latest_evidence")),
        "hard_constraints": _dedupe_text_items(_list_value(raw.get("hard_constraints")), limit=8),
    }


def _normalize_question_understanding(value: object, question: str) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    understanding = dict(_DEFAULT_QUESTION_UNDERSTANDING)
    understanding["user_goal"] = " ".join(str(raw.get("user_goal") or question).split()).strip()
    understanding["result_shape"] = _normalize_result_shape(raw.get("result_shape"))
    understanding["search_intent"] = " ".join(
        str(raw.get("search_intent") or _DEFAULT_QUESTION_UNDERSTANDING["search_intent"]).split()
    ).strip()
    understanding["coverage_expectation"] = _normalize_coverage_expectation(raw.get("coverage_expectation"))
    understanding["organization_strategy"] = _normalize_organization_strategy(raw.get("organization_strategy"))
    understanding["freshness_sensitivity"] = _normalize_freshness_sensitivity(raw.get("freshness_sensitivity"))
    understanding["evidence_priority"] = _dedupe_text_items(_list_value(raw.get("evidence_priority")), limit=8)
    understanding["known_risks"] = _dedupe_text_items(_list_value(raw.get("known_risks")), limit=8)
    return understanding


def _normalize_contextual_question(value: object, question: str) -> ContextualQuestion:
    raw = value if isinstance(value, dict) else {}
    resolved = _text(raw.get("resolved_question")) or question
    return ContextualQuestion(
        is_followup=bool(raw.get("is_followup", False)),
        resolved_question=resolved,
        current_topic=_text(raw.get("current_topic")),
        turn_operation=_normalize_turn_operation(raw.get("turn_operation"), is_followup=bool(raw.get("is_followup", False))),
        reuse_strategy=_normalize_reuse_strategy(raw.get("reuse_strategy"), is_followup=bool(raw.get("is_followup", False))),
        target_entity_type=_normalize_target_entity_type(raw.get("target_entity_type")),
        required_attributes=_dedupe_text_items(_list_value(raw.get("required_attributes")), limit=8),
        candidate_filters=_dedupe_text_items(_list_value(raw.get("candidate_filters")), limit=8),
        active_entities=_dedupe_text_items(_list_value(raw.get("active_entities")), limit=10),
        active_constraints=_dedupe_text_items(_list_value(raw.get("active_constraints")), limit=10),
        dropped_context=_dedupe_text_items(_list_value(raw.get("dropped_context")), limit=10),
        retrieval_instruction=_text(raw.get("retrieval_instruction")),
        answer_style_instruction=_text(raw.get("answer_style_instruction")),
        session_summary=_text(raw.get("session_summary")),
        open_questions=_dedupe_text_items(_list_value(raw.get("open_questions")), limit=10),
    )


def _normalize_entity_synthesis(value: object, contextual_question: ContextualQuestion) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    return {
        "entity_type": _normalize_target_entity_type(raw.get("entity_type") or contextual_question.target_entity_type),
        "entity_set": _normalize_entity_set(raw.get("entity_set")),
        "entity_merge_notes": _dedupe_text_items(_list_value(raw.get("entity_merge_notes")), limit=10),
        "missing_attributes": _dedupe_text_items(_list_value(raw.get("missing_attributes")), limit=10),
        "insufficient_entities": _dedupe_text_items(_list_value(raw.get("insufficient_entities")), limit=10),
    }


def _contextual_question_to_dict(value: ContextualQuestion) -> dict[str, object]:
    return {
        "is_followup": value.is_followup,
        "resolved_question": value.resolved_question,
        "current_topic": value.current_topic,
        "turn_operation": value.turn_operation,
        "reuse_strategy": value.reuse_strategy,
        "target_entity_type": value.target_entity_type,
        "required_attributes": value.required_attributes,
        "candidate_filters": value.candidate_filters,
        "active_entities": value.active_entities,
        "active_constraints": value.active_constraints,
        "dropped_context": value.dropped_context,
        "retrieval_instruction": value.retrieval_instruction,
        "answer_style_instruction": value.answer_style_instruction,
        "session_summary": value.session_summary,
        "open_questions": value.open_questions,
    }


def _normalize_entity_set(value: object) -> list[dict[str, object]]:
    entities: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in _list_value(value):
        if not isinstance(item, dict):
            continue
        entity_name = _text(item.get("entity_name"))
        if not entity_name:
            continue
        entity_id = _text(item.get("entity_id")) or entity_name
        if entity_id in seen:
            continue
        seen.add(entity_id)
        attributes = item.get("attributes")
        entities.append(
            {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "entity_type": _normalize_target_entity_type(item.get("entity_type")),
                "aliases": _dedupe_text_items(_list_value(item.get("aliases")), limit=8),
                "attributes": attributes if isinstance(attributes, dict) else {},
                "evidence_urls": _dedupe_text_items(_list_value(item.get("evidence_urls")), limit=10),
                "evidence_ids": _dedupe_text_items(_list_value(item.get("evidence_ids")), limit=12),
                "confidence": _normalize_confidence(item.get("confidence")),
            }
        )
        if len(entities) >= 20:
            break
    return entities


def _normalize_topic_choices(
    items: list[object],
    evidence: list[dict[str, object]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    allowed = {str(item.get("url", "")) for item in evidence}
    choices: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url or url not in allowed or url in seen:
            continue
        seen.add(url)
        choices.append({"url": url, "reason": " ".join(str(item.get("reason", "")).split()).strip()})
        if len(choices) >= limit:
            break
    return choices


def _normalize_question_shape(value: object) -> str:
    shape = str(value or "").strip()
    allowed = {"single_fact", "procedure", "enumeration", "comparison", "recommendation", "time_sensitive_rule"}
    return shape if shape in allowed else "single_fact"


def _normalize_result_shape(value: object) -> str:
    shape = str(value or "").strip()
    allowed = {"single_best", "actionable_steps", "broad_options", "current_rule", "compare_options"}
    return shape if shape in allowed else "single_best"


def _normalize_turn_operation(value: object, *, is_followup: bool) -> str:
    operation = str(value or "").strip()
    allowed = {"new_topic", "refine_scope", "filter_candidates", "compare_candidates", "fill_missing_attributes"}
    if operation in allowed:
        return operation
    return "refine_scope" if is_followup else "new_topic"


def _normalize_reuse_strategy(value: object, *, is_followup: bool) -> str:
    strategy = str(value or "").strip()
    allowed = {"reuse_only", "reuse_then_expand", "full_refresh"}
    if strategy in allowed:
        return strategy
    return "reuse_then_expand" if is_followup else "full_refresh"


def _normalize_target_entity_type(value: object) -> str:
    entity_type = str(value or "").strip()
    allowed = {"person", "lab", "course", "location", "department", "unknown"}
    return entity_type if entity_type in allowed else "unknown"


def _normalize_confidence(value: object) -> str:
    confidence = str(value or "").strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else "medium"


def _normalize_coverage_expectation(value: object) -> str:
    level = str(value or "").strip()
    return level if level in {"narrow", "medium", "wide"} else "narrow"


def _normalize_organization_strategy(value: object) -> str:
    strategy = str(value or "").strip()
    allowed = {"flat_summary", "by_steps", "by_entities", "by_current_vs_history", "by_options"}
    return strategy if strategy in allowed else "flat_summary"


def _normalize_freshness_sensitivity(value: object) -> str:
    level = str(value or "").strip()
    return level if level in {"low", "medium", "high"} else "medium"


def _normalize_topic_scores(
    items: list[object],
    evidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    allowed = {str(item.get("url", "")) for item in evidence}
    scores: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url or url not in allowed or url in seen:
            continue
        seen.add(url)
        try:
            expand_score = max(0.0, min(1.0, float(item.get("expand_score", 0.0))))
        except (TypeError, ValueError):
            expand_score = 0.0
        expected_raw = item.get("expected_value")
        expected = expected_raw if isinstance(expected_raw, dict) else {}
        scores.append(
            {
                "url": url,
                "expand_score": round(expand_score, 3),
                "reason": " ".join(str(item.get("reason", "")).split()).strip(),
                "expected_value": {
                    "entity_discovery": _normalize_expected_level(expected.get("entity_discovery")),
                    "rule_detail": _normalize_expected_level(expected.get("rule_detail")),
                    "actionability": _normalize_expected_level(expected.get("actionability")),
                    "freshness_signal": _normalize_expected_level(expected.get("freshness_signal")),
                    "complementarity": _normalize_expected_level(expected.get("complementarity")),
                },
            }
        )
    scores.sort(key=lambda item: float(item["expand_score"]), reverse=True)
    return scores


def _normalize_candidate_profiles(
    items: list[object],
    evidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    allowed_urls = {str(item.get("url", "")) for item in evidence}
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = " ".join(str(item.get("url", "")).split()).strip()
        if not url or url not in allowed_urls or url in seen:
            continue
        seen.add(url)
        normalized.append(
            {
                "url": url,
                "primary_object": " ".join(str(item.get("primary_object", "")).split()).strip(),
                "object_kind": _normalize_object_kind(item.get("object_kind")),
                "scope": _normalize_scope(item.get("scope")),
                "coverage_tags": _dedupe_text_items(_list_value(item.get("coverage_tags")), limit=8),
                "redundant_with": " ".join(str(item.get("redundant_with", "")).split()).strip(),
                "reason": " ".join(str(item.get("reason", "")).split()).strip(),
            }
        )
    return normalized


def _normalize_object_kind(value: object) -> str:
    kind = str(value or "").strip()
    allowed = {"person", "lab", "team", "department", "place", "course", "service", "unknown"}
    return kind if kind in allowed else "unknown"


def _normalize_scope(value: object) -> str:
    scope = str(value or "").strip()
    allowed = {"comprehensive", "specialized", "single_case", "discussion", "unknown"}
    return scope if scope in allowed else "unknown"


def _normalize_expected_level(value: object) -> str:
    level = str(value or "").strip().lower()
    return level if level in {"high", "medium", "low"} else "low"


def _normalize_structured_group(items: list[object], *, kind: str) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in items[:12]:
        if not isinstance(item, dict):
            continue
        if kind == "entity":
            name = " ".join(str(item.get("name", "")).split()).strip()
            if not name:
                continue
            normalized.append(
                {
                    "name": name,
                    "kind": " ".join(str(item.get("kind", "")).split()).strip(),
                    "evidence_url": " ".join(str(item.get("evidence_url", "")).split()).strip(),
                    "facts": _dedupe_text_items(_list_value(item.get("facts")), limit=8),
                    "freshness_level": _normalize_freshness_level(item.get("freshness_level")),
                    "evidence_time": " ".join(str(item.get("evidence_time", "")).split()).strip(),
                }
            )
            continue
        key_name = {
            "action": "action",
            "rule": "rule",
            "time": "constraint",
            "uncertainty": "uncertainty",
        }[kind]
        text = " ".join(str(item.get(key_name, "")).split()).strip()
        if not text:
            continue
        normalized.append(
            {
                key_name: text,
                "evidence_url": " ".join(str(item.get("evidence_url", "")).split()).strip(),
                "details": _dedupe_text_items(_list_value(item.get("details")), limit=6),
                "freshness_level": _normalize_freshness_level(item.get("freshness_level")),
                "evidence_time": " ".join(str(item.get("evidence_time", "")).split()).strip(),
            }
        )
    return normalized


def _normalize_structured_summary_group(items: list[object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        claim = " ".join(str(item.get("claim", "")).split()).strip()
        if not claim:
            continue
        normalized.append(
            {
                "claim": claim,
                "evidence_url": " ".join(str(item.get("evidence_url", "")).split()).strip(),
                "details": _dedupe_text_items(_list_value(item.get("details")), limit=6),
            }
        )
    return normalized


def _normalize_freshness_level(value: object) -> str:
    level = str(value or "").strip()
    allowed = {"current_cycle", "stable_practice", "historical_or_uncertain"}
    return level if level in allowed else "historical_or_uncertain"


def _dedupe_text_items(items: list[object], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item).split()).strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


def _strip_json_fence(value: str) -> str:
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return value


def _post_chat_completion(config: LLMConfig, payload: dict[str, object]) -> dict[str, object]:
    url = f"{config.api_base}/chat/completions"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
