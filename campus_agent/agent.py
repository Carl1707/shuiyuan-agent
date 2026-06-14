from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import replace

from campus_agent.chunking import chunk_document
from campus_agent.retrieval import retrieve, tokenize
from campus_agent.llm import (
    EvidenceAssessment,
    LLMError,
    ShuiyuanSearchPlan,
    assess_shuiyuan_evidence,
    build_evidence_ledger,
    extract_structured_community_evidence,
    generate_shuiyuan_search_plan,
    generate_verified_answer,
)
from campus_agent.models import Chunk, Document, RetrievalResult
from campus_agent.tools import CommunityDocument, CommunityPost, CommunitySearchResult, CommunitySearchTool

_QUESTION_STOP_TERMS = {
    "如何",
    "怎么",
    "怎样",
    "哪些",
    "哪个",
    "哪里",
    "几时",
    "多少",
    "是否",
    "可以",
    "需要",
    "请问",
    "想问",
    "一下",
    "一下子",
    "有谁",
    "有人",
    "什么",
    "时候",
}
_BODY_FETCH_MIN_RELEVANCE = 0.2
_PER_QUERY_SEARCH_LIMIT = 8
_BODY_FETCH_CANDIDATE_LIMIT = 5
_BODY_CACHE_TTL_SECONDS = 600
_BODY_CHUNK_MAX_CHARS = 700
_BODY_CHUNK_OVERLAP_CHARS = 100
_BODY_RESULT_PER_TOPIC_LIMIT = 2
_BODY_RAG_CHUNK_LIMIT = 5
_WIKI_HINT_TERMS = ("wiki", "指南", "教程", "攻略", "手册")
_BODY_CACHE: dict[str, tuple[float, CommunityDocument | None]] = {}
_MAX_SEARCH_QUERIES = 7
_LEDGER_CANDIDATE_LIMIT = 30
_LATEST_QUESTION_TERMS = ("最新", "今年", "当年", "现在", "当前", "最近", "截止", "要求", "流程", "时间", "条件")


def answer_question(
    question: str,
    *,
    question_type: str = "",
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    use_query_rewriting: bool = True,
    use_rerank: bool = True,
    use_body_rag: bool = True,
    require_highlight: bool = True,
    community_tool: CommunitySearchTool | None = None,
    answer_backend: str = "extractive",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_api_base: str | None = None,
    llm_timeout_seconds: int | None = None,
    progress_callback=None,
    **_legacy_options: object,
) -> dict[str, object]:
    """Answer from live Shuiyuan results using an LLM-driven evidence loop."""

    _emit_progress(progress_callback, "query_planning", "正在生成桥接搜索计划")
    query_rewrite_backend = "question-only"
    search_plan = _fallback_search_plan(question)
    if use_query_rewriting:
        try:
            search_plan = generate_shuiyuan_search_plan(
                question=question,
                question_type=question_type,
                max_queries=15,
                model=llm_model,
                api_key=llm_api_key,
                api_base=llm_api_base,
                timeout_seconds=llm_timeout_seconds,
            )
            query_rewrite_backend = "llm"
        except LLMError:
            search_plan = _fallback_search_plan(question)
            query_rewrite_backend = "question-fallback"
    query_details = search_plan.query_details or [
        {"query": query, "purpose": ""} for query in (search_plan.queries or [question])
    ]
    selected_query_details = search_plan.selected_query_details or query_details[:_MAX_SEARCH_QUERIES]
    selected_query_details = _dedupe_query_details(selected_query_details)[:_MAX_SEARCH_QUERIES]
    executed_queries = [item["query"] for item in selected_query_details]

    _emit_progress(progress_callback, "searching", "正在执行 LLM 筛选后的 Shuiyuan 查询")
    community_results = _execute_search_batch(
        community_tool=community_tool,
        queries=executed_queries,
        progress_callback=progress_callback,
        completed_count=0,
        planned_count=len(executed_queries),
    )
    observed_wait_seconds = 0
    observed_wait_seconds = max(observed_wait_seconds, _tool_wait_seconds(community_tool))
    ranked_results = rank_community_results(
        question,
        community_results,
        answer_contract=search_plan.answer_contract,
    )
    _emit_progress(progress_callback, "evidence_audit", "正在区分直接答案、辅助信息和背景")
    try:
        latest_assessment = assess_shuiyuan_evidence(
            question=question,
            answer_contract=search_plan.answer_contract,
            community_results=ranked_results,
            model=llm_model,
            api_key=llm_api_key,
            api_base=llm_api_base,
            timeout_seconds=llm_timeout_seconds,
        )
    except LLMError:
        latest_assessment = _fallback_assessment(search_plan.answer_contract)
    ranked_results = _rank_by_evidence_roles(ranked_results, latest_assessment.evidence_roles)
    coverage_assessments = [_assessment_to_dict(latest_assessment, 1)]
    search_batches = [
        {
            "batch": 1,
            "queries": selected_query_details,
            "new_result_count": len(community_results),
            "deduped_result_count": len(ranked_results),
        }
    ]

    if use_body_rag:
        _set_tool_progress_context(
            community_tool,
            lambda message: _emit_progress(progress_callback, "body_rag", message),
        )
        _emit_progress(progress_callback, "body_rag", "正在按证据缺口展开帖子回复")
        community_evidence, ledger_evidence, enriched_results, expanded_topics, expanded_topic_docs = _retrieve_community_evidence(
            question=question,
            ranked_results=ranked_results,
            community_tool=community_tool,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            bridges=search_plan.bridges,
            topics_to_expand=latest_assessment.topics_to_expand,
            question_shape=latest_assessment.question_shape,
        )
        observed_wait_seconds = max(observed_wait_seconds, _tool_wait_seconds(community_tool))
    else:
        community_evidence = _search_result_evidence(ranked_results, question, top_k=max(top_k, 5))
        ledger_evidence = community_evidence
        enriched_results = ranked_results[:top_k]
        expanded_topics = []
        expanded_topic_docs = []

    _emit_progress(progress_callback, "body_rag", "正在从已展开正文中抽取结构化证据")
    try:
        structured_evidence = extract_structured_community_evidence(
            question=question,
            question_shape=latest_assessment.question_shape,
            answer_contract=search_plan.answer_contract,
            expanded_topics=expanded_topic_docs,
            model=llm_model,
            api_key=llm_api_key,
            api_base=llm_api_base,
            timeout_seconds=llm_timeout_seconds,
        )
    except LLMError:
        structured_evidence = _fallback_structured_evidence()

    evidence_items = _evidence_items(ledger_evidence, ranked_results)
    _emit_progress(progress_callback, "fact_ledger", "正在从候选证据中整理可支持事实")
    try:
        evidence_ledger = build_evidence_ledger(
            question=question,
            question_shape=latest_assessment.question_shape,
            answer_contract=search_plan.answer_contract,
            evidence_items=evidence_items,
            structured_evidence=structured_evidence,
            model=llm_model,
            api_key=llm_api_key,
            api_base=llm_api_base,
            timeout_seconds=llm_timeout_seconds,
        )
    except LLMError:
        evidence_ledger = _fallback_evidence_ledger(evidence_items)

    _emit_progress(progress_callback, "generating", "正在校验证据并生成回答")
    if answer_backend == "llm":
        try:
            answer = generate_verified_answer(
                question=question,
                question_shape=latest_assessment.question_shape,
                answer_contract=search_plan.answer_contract,
                evidence_ledger=evidence_ledger,
                evidence_items=evidence_items,
                structured_evidence=structured_evidence,
                model=llm_model,
                api_key=llm_api_key,
                api_base=llm_api_base,
                timeout_seconds=llm_timeout_seconds,
            )
        except LLMError:
            answer = _generate_extractive_answer(enriched_results)
    else:
        answer = _generate_extractive_answer(enriched_results)
    return {
        "question": question,
        "queries": executed_queries,
        "search_plan": {
            "intent": search_plan.intent,
            "bridges": search_plan.bridges,
            "candidate_queries": search_plan.queries,
            "executed_queries": executed_queries,
            "query_details": query_details,
            "selected_query_details": selected_query_details,
            "rejected_query_details": search_plan.rejected_query_details,
            "answer_contract": search_plan.answer_contract,
            "question_shape": latest_assessment.question_shape,
            "topic_scores": latest_assessment.topic_scores,
        },
        "answer_contract": search_plan.answer_contract,
        "question_shape": latest_assessment.question_shape,
        "topic_scores": latest_assessment.topic_scores,
        "query_details": query_details,
        "selected_query_details": selected_query_details,
        "rejected_query_details": search_plan.rejected_query_details,
        "search_batches": search_batches,
        "coverage_assessments": coverage_assessments,
        "expanded_topics": expanded_topics,
        "structured_evidence": structured_evidence,
        "evidence_ledger": evidence_ledger,
        "answer": answer,
        "results": community_evidence,
        "community_results": enriched_results,
        "query_rewrite_backend": query_rewrite_backend,
        "body_rag_used": use_body_rag,
        "observed_wait_seconds": observed_wait_seconds,
        "llm_call_count": 4 if answer_backend == "llm" else 3,
        "search_request_count": len(executed_queries) + len(expanded_topics),
    }


def _emit_progress(progress_callback, step: str, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(step, message)


def _tool_wait_seconds(community_tool: CommunitySearchTool | None) -> int:
    if community_tool is None:
        return 0
    return int(getattr(community_tool, "total_rate_limit_wait_seconds", 0) or 0)


def _set_tool_progress_context(
    community_tool: CommunitySearchTool | None,
    callback,
) -> None:
    if community_tool is None:
        return
    setter = getattr(community_tool, "set_progress_callback", None)
    if callable(setter):
        setter(callback)


def _fallback_search_plan(question: str) -> ShuiyuanSearchPlan:
    contract = {
        "user_need": question,
        "required_facts": ["能够直接回答用户问题的社区事实"],
        "helpful_facts": [],
        "insufficient_evidence": ["只与主题相关但不能回答问题的内容"],
        "requires_latest_evidence": False,
        "hard_constraints": [],
    }
    return ShuiyuanSearchPlan(
        intent="校园事务搜索",
        bridges=[],
        queries=[question],
        answer_contract=contract,
        query_details=[{"query": question, "purpose": "直接搜索用户原问题"}],
    )


def _fallback_assessment(answer_contract: dict[str, object]) -> EvidenceAssessment:
    required = answer_contract.get("required_facts", [])
    return EvidenceAssessment(
        question_shape="single_fact",
        can_answer=False,
        covered_requirements=[],
        missing_requirements=[str(item) for item in required] if isinstance(required, list) else [],
        next_queries=[],
        topics_to_expand=[],
        topic_scores=[],
        summary="LLM 证据审计不可用，继续使用已有候选证据。",
    )


def _assessment_to_dict(assessment: EvidenceAssessment, batch_number: int) -> dict[str, object]:
    return {
        "after_batch": batch_number,
        "can_answer": assessment.can_answer,
        "covered_requirements": assessment.covered_requirements,
        "missing_requirements": assessment.missing_requirements,
        "next_queries": assessment.next_queries,
        "question_shape": assessment.question_shape,
        "topics_to_expand": assessment.topics_to_expand,
        "topic_scores": assessment.topic_scores,
        "summary": assessment.summary,
        "evidence_roles": assessment.evidence_roles,
    }


def _dedupe_query_details(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        query = " ".join(str(item.get("query", "")).split()).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        deduped.append(
            {
                "query": query,
                "purpose": " ".join(str(item.get("purpose", "")).split()).strip(),
                "lane": " ".join(str(item.get("lane", "direct")).split()).strip() or "direct",
            }
        )
    return deduped


def _merge_query_details(
    current: list[dict[str, str]],
    additional: list[dict[str, str]],
) -> list[dict[str, str]]:
    return _dedupe_query_details([*current, *additional])[:_MAX_SEARCH_QUERIES]


def _take_query_batch(
    query_queue: list[dict[str, str]],
    executed_queries: list[str],
    *,
    limit: int,
) -> list[dict[str, str]]:
    executed = set(executed_queries)
    return [item for item in query_queue if item["query"] not in executed][:limit]


def _execute_search_batch(
    *,
    community_tool: CommunitySearchTool | None,
    queries: list[str],
    progress_callback,
    completed_count: int,
    planned_count: int,
) -> list[CommunitySearchResult]:
    if community_tool is None:
        return []
    results: list[CommunitySearchResult] = []
    for offset, query in enumerate(queries, start=1):
        _set_tool_progress_context(
            community_tool,
            lambda message: _emit_progress(progress_callback, "searching", message),
        )
        _emit_progress(
            progress_callback,
            "searching",
            f"正在搜索 Shuiyuan ({completed_count + offset}/{max(planned_count, completed_count + len(queries))})",
        )
        results.extend(community_tool.search(query, limit=_PER_QUERY_SEARCH_LIMIT))
    return results


def rank_community_results(
    question: str,
    results: list[CommunitySearchResult],
    answer_contract: dict[str, object] | None = None,
) -> list[CommunitySearchResult]:
    """Deduplicate and rank Shuiyuan posts using Shuiyuan-native evidence signals."""

    deduped: list[CommunitySearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        deduped.append(result)

    latest_sensitive = _requires_latest_evidence(question, answer_contract)
    support_counts = _compute_support_counts(deduped)
    for result in deduped:
        support_count = support_counts.get(result.url, 0)
        result.support_count = support_count
        result.relevance_score = round(
            _text_relevance(question, result.title, result.snippet)
            + _community_layer_bonus(
                question=question,
                result=result,
                latest_sensitive=latest_sensitive,
                support_count=support_count,
            ),
            6,
        )
    deduped.sort(key=lambda item: item.relevance_score, reverse=True)
    return deduped


def _rank_by_evidence_roles(
    ranked_results: list[CommunitySearchResult],
    evidence_roles: list[dict[str, str]],
) -> list[CommunitySearchResult]:
    role_priority = {
        "direct_answer": 4,
        "useful_support": 3,
        "background_only": 2,
        "irrelevant": 1,
    }
    roles = {item["url"]: item["role"] for item in evidence_roles}
    return sorted(
        ranked_results,
        key=lambda item: (role_priority.get(roles.get(item.url, ""), 0), item.relevance_score),
        reverse=True,
    )


def _retrieve_community_evidence(
    *,
    question: str,
    ranked_results: list[CommunitySearchResult],
    community_tool: CommunitySearchTool | None,
    retrieval_mode: str,
    top_k: int,
    bridges: list[str],
    topics_to_expand: list[dict[str, str]],
    question_shape: str,
) -> tuple[list[RetrievalResult], list[RetrievalResult], list[CommunitySearchResult], list[dict[str, str]], list[dict[str, object]]]:
    if community_tool is None or not ranked_results:
        evidence = _search_result_evidence(ranked_results, question, top_k=max(top_k, 5))
        return evidence, evidence, ranked_results[:top_k], [], []

    fetch_candidates = _select_body_fetch_candidates(ranked_results, topics_to_expand, question_shape=question_shape)
    if not fetch_candidates:
        evidence = _search_result_evidence(ranked_results, question, top_k=max(top_k, 5))
        return evidence, evidence, ranked_results[:top_k], [], []

    search_evidence = _search_result_evidence(ranked_results, question, top_k=20)
    fetched_urls = {result.url for result in fetch_candidates}
    candidate_chunks = [
        hit.chunk
        for hit in search_evidence
        if str(hit.chunk.metadata.get("topic_url") or "") not in fetched_urls
    ]
    enriched_by_url: dict[str, CommunitySearchResult] = {}
    expanded_topics: list[dict[str, str]] = []
    expanded_topic_docs: list[dict[str, object]] = []
    for result in fetch_candidates:
        topic = _fetch_topic_with_cache(community_tool, result.url)
        if topic is None or not topic.text.strip():
            continue
        topic_chunks = _chunk_topic_document(topic)
        if not topic_chunks:
            enriched_by_url[result.url] = replace(result, body_loaded=True)
            continue
        candidate_chunks.extend(topic_chunks)
        enriched_by_url[result.url] = replace(result, body_loaded=True)
        expanded_topics.append(
            {
                "url": topic.url,
                "title": topic.title,
                "reason": next(
                    (item.get("reason", "") for item in topics_to_expand if item.get("url") == result.url),
                    "作为高相关候选帖子补充正文证据",
                ),
                "post_count": len(topic.posts),
                "expand_score": next(
                    (item.get("expand_score", 0.0) for item in topics_to_expand if item.get("url") == result.url),
                    0.0,
                ),
                "expected_value": next(
                    (item.get("expected_value", {}) for item in topics_to_expand if item.get("url") == result.url),
                    {},
                ),
            }
        )
        expanded_topic_docs.append(
            {
                "url": topic.url,
                "title": topic.title,
                "updated_at": topic.updated_at,
                "created_at": topic.created_at,
                "tags": topic.tags,
                "text": topic.text,
                "posts": [
                    {
                        "post_number": post.post_number,
                        "username": post.username,
                        "created_at": post.created_at,
                        "updated_at": post.updated_at,
                        "text": post.text,
                    }
                    for post in topic.posts
                ],
            }
        )

    evidence_hits = _retrieve_cross_topic_evidence(
        question=question,
        chunks=candidate_chunks,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        bridges=bridges,
    )
    for hit in evidence_hits:
        topic_url = str(hit.chunk.metadata.get("topic_url") or "")
        if not topic_url:
            continue
        excerpt = (hit.highlights[0] if hit.highlights else hit.chunk.text[:320]).replace("\n", " ")
        existing = enriched_by_url.get(topic_url)
        if existing is None:
            continue
        if existing.snippet == excerpt:
            continue
        enriched_by_url[topic_url] = replace(existing, snippet=excerpt, body_loaded=True)

    final_results = []
    seen_urls: set[str] = set()
    for result in ranked_results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        final_results.append(enriched_by_url.get(result.url, result))
        if len(final_results) >= top_k:
            break
    ledger_evidence = [
        RetrievalResult(chunk=chunk, score=0.0, highlights=[chunk.text])
        for chunk in candidate_chunks
    ]
    return evidence_hits, ledger_evidence, final_results, expanded_topics, expanded_topic_docs


def _text_relevance(question: str, title: str, snippet: str) -> float:
    query_terms = set(tokenize(question))
    query_grams = _character_ngrams(question)
    if not query_terms and not query_grams:
        return 0.0

    title_terms = set(tokenize(title))
    body_terms = set(tokenize(snippet))
    title_grams = _character_ngrams(title)
    body_grams = _character_ngrams(snippet)
    term_score = 2.0 * _overlap(query_terms, title_terms) + _overlap(query_terms, body_terms)
    gram_score = 2.0 * _overlap(query_grams, title_grams) + _overlap(query_grams, body_grams)
    return round(0.45 * term_score + 0.55 * gram_score, 6)


def _result_matches_question_core_terms(
    result: CommunitySearchResult,
    core_terms: set[str],
) -> bool:
    if not core_terms:
        return bool(result.title.strip() or result.snippet.strip())
    return _contains_core_term(result.title, core_terms) or _contains_core_term(result.snippet, core_terms)


def _select_body_fetch_candidates(
    ranked_results: list[CommunitySearchResult],
    topics_to_expand: list[dict[str, str]],
    *,
    question_shape: str,
) -> list[CommunitySearchResult]:
    by_url = {result.url: result for result in ranked_results}
    selected = [
        by_url[item["url"]]
        for item in topics_to_expand
        if item.get("url") in by_url
    ]
    if selected:
        return _diversify_fetch_candidates(selected, budget=_expansion_budget(question_shape))
    return [
        result
        for result in ranked_results
        if result.relevance_score >= _BODY_FETCH_MIN_RELEVANCE
    ][:_expansion_budget(question_shape)]


def _requires_latest_evidence(question: str, answer_contract: dict[str, object] | None) -> bool:
    if bool((answer_contract or {}).get("requires_latest_evidence")):
        return True
    normalized = question.lower()
    return any(term in normalized for term in _LATEST_QUESTION_TERMS)


def _compute_support_counts(results: list[CommunitySearchResult]) -> dict[str, int]:
    per_result_terms: dict[str, set[str]] = {}
    term_counts: Counter[str] = Counter()
    for result in results:
        terms = _support_terms(result.title, result.snippet)
        per_result_terms[result.url] = terms
        term_counts.update(terms)
    return {
        url: sum(1 for term in terms if term_counts[term] >= 2)
        for url, terms in per_result_terms.items()
    }


def _support_terms(title: str, snippet: str) -> set[str]:
    return {
        term
        for term in _extract_core_terms(f"{title} {snippet}")
        if len(term) >= 2 and not term.isdigit()
    }


def _community_layer_bonus(
    *,
    question: str,
    result: CommunitySearchResult,
    latest_sensitive: bool,
    support_count: int,
) -> float:
    bonus = 0.0
    if result.is_wiki:
        bonus += 0.14
    bonus += _temporal_bonus(result, latest_sensitive=latest_sensitive)
    bonus += min(support_count, 4) * 0.025
    if latest_sensitive and _looks_like_background_post(result, question):
        bonus -= 0.06
    return bonus


def _temporal_bonus(result: CommunitySearchResult, *, latest_sensitive: bool) -> float:
    year = _extract_year(result.updated_at) or _extract_year(result.created_at)
    if year <= 0:
        return 0.0
    current_year = time.localtime().tm_year
    delta = current_year - year
    if latest_sensitive:
        if delta <= 0:
            return 0.2
        if delta == 1:
            return 0.12
        if delta == 2:
            return 0.04
        if delta >= 4:
            return -0.08
        return -0.02
    if delta <= 0:
        return 0.06
    if delta == 1:
        return 0.03
    if delta >= 5:
        return -0.02
    return 0.0


def _extract_year(value: str) -> int:
    match = re.search(r"\b(20\d{2})\b", value or "")
    if not match:
        return 0
    return int(match.group(1))


def _looks_like_background_post(result: CommunitySearchResult, question: str) -> bool:
    normalized = f"{result.title} {result.snippet}".lower()
    if result.is_wiki:
        return False
    if any(term in normalized for term in ("闲聊", "日常", "水楼", "吐槽", "灌水")):
        return True
    year = _extract_year(result.updated_at) or _extract_year(result.created_at)
    return year > 0 and year < time.localtime().tm_year - 2 and _text_relevance(question, result.title, result.snippet) < 0.18


def _chunk_topic_document(topic: CommunityDocument) -> list[Chunk]:
    if topic.posts:
        chunks: list[Chunk] = []
        for post in topic.posts:
            chunks.extend(_chunk_topic_post(topic, post))
        return chunks
    document = Document(
        id=f"shuiyuan::{topic.url}",
        title=topic.title,
        text=topic.text,
        uri=topic.url,
        content_type="text/plain",
        metadata={
            "updated_at": topic.updated_at,
            "created_at": topic.created_at,
            "evidence_origin": "shuiyuan_topic_body",
            "topic_url": topic.url,
            "topic_title": topic.title,
            "topic_tags": topic.tags,
            "topic_is_wiki": topic.is_wiki,
            "topic_has_solution": topic.has_solution,
            "solution_post_number": topic.solution_post_number,
        },
    )
    return chunk_document(
        document,
        max_chars=_BODY_CHUNK_MAX_CHARS,
        overlap_chars=_BODY_CHUNK_OVERLAP_CHARS,
    )


def _chunk_topic_post(topic: CommunityDocument, post: CommunityPost) -> list[Chunk]:
    post_url = f"{topic.url}/{post.post_number}"
    post_text = f"主题：{topic.title}\n内容：{post.text}"
    document = Document(
        id=f"shuiyuan::{topic.url}::post::{post.post_number}",
        title=topic.title,
        text=post_text,
        uri=post_url,
        content_type="text/plain",
        metadata={
            "updated_at": post.updated_at,
            "created_at": post.created_at,
            "evidence_origin": "shuiyuan_topic_reply",
            "topic_url": topic.url,
            "post_url": post_url,
            "post_number": post.post_number,
            "username": post.username,
            "topic_title": topic.title,
            "topic_tags": topic.tags,
            "topic_is_wiki": topic.is_wiki,
            "topic_has_solution": topic.has_solution,
            "solution_post_number": topic.solution_post_number,
            "is_solution": post.is_solution,
        },
    )
    return chunk_document(
        document,
        max_chars=_BODY_CHUNK_MAX_CHARS,
        overlap_chars=_BODY_CHUNK_OVERLAP_CHARS,
    )


def _search_result_evidence(
    ranked_results: list[CommunitySearchResult],
    question: str,
    *,
    top_k: int,
) -> list[RetrievalResult]:
    evidence: list[RetrievalResult] = []
    for index, item in enumerate(ranked_results[:top_k], start=1):
        chunk = Chunk(
            id=f"shuiyuan-search::{index}::{item.url}",
            document_id=f"shuiyuan-search::{item.url}",
            text=item.snippet,
            metadata={
                "document_title": item.title,
                "document_uri": item.url,
                "topic_title": item.title,
                "topic_url": item.url,
                "evidence_origin": "shuiyuan_search_excerpt",
                "topic_is_wiki": item.is_wiki,
                "topic_has_solution": item.has_solution,
                "solution_post_number": item.solution_post_number,
                "topic_tags": item.tags,
                "support_count": item.support_count,
            },
        )
        evidence.append(
            RetrievalResult(
                chunk=chunk,
                score=item.relevance_score or _text_relevance(question, item.title, item.snippet),
                highlights=[item.snippet],
            )
        )
    return evidence


def _retrieve_cross_topic_evidence(
    *,
    question: str,
    chunks: list[Chunk],
    retrieval_mode: str,
    top_k: int,
    bridges: list[str],
) -> list[RetrievalResult]:
    if not chunks:
        return []
    query = question
    if bridges:
        query = f"{question}\n桥接概念：{' '.join(bridges[:6])}"
    raw_hits = retrieve(
        chunks,
        query,
        top_k=min(len(chunks), max(top_k * 8, _BODY_RAG_CHUNK_LIMIT * 3)),
        mode=retrieval_mode,
    )
    rescored: list[RetrievalResult] = []
    for hit in raw_hits:
        excerpt = hit.highlights[0] if hit.highlights else hit.chunk.text
        bridge_bonus = _bridge_match_bonus(excerpt, bridges)
        signal_bonus = _chunk_signal_bonus(hit.chunk)
        rescored.append(
            RetrievalResult(
                chunk=hit.chunk,
                score=round(hit.score + bridge_bonus + signal_bonus, 6),
                highlights=hit.highlights,
            )
        )
    rescored.sort(key=lambda item: item.score, reverse=True)
    return _dedupe_evidence_hits(rescored, limit=max(top_k, _BODY_RAG_CHUNK_LIMIT))


def _dedupe_evidence_hits(
    hits: list[RetrievalResult],
    *,
    limit: int,
) -> list[RetrievalResult]:
    deduped: list[RetrievalResult] = []
    seen_texts: set[str] = set()
    per_topic: dict[str, int] = {}
    for hit in hits:
        topic_url = str(hit.chunk.metadata.get("topic_url") or "")
        normalized = re.sub(r"\W+", "", hit.chunk.text).lower()
        if not normalized or normalized in seen_texts:
            continue
        if topic_url and per_topic.get(topic_url, 0) >= _BODY_RESULT_PER_TOPIC_LIMIT:
            continue
        seen_texts.add(normalized)
        if topic_url:
            per_topic[topic_url] = per_topic.get(topic_url, 0) + 1
        deduped.append(hit)
        if len(deduped) >= limit:
            break
    return deduped


def _fetch_topic_with_cache(
    community_tool: CommunitySearchTool,
    url: str,
) -> CommunityDocument | None:
    cached = _BODY_CACHE.get(url)
    now = time.time()
    if cached is not None and now - cached[0] <= _BODY_CACHE_TTL_SECONDS:
        return cached[1]
    topic = community_tool.fetch_topic(url)
    _BODY_CACHE[url] = (now, topic)
    return topic


def _bridge_match_bonus(text: str, bridges: list[str]) -> float:
    if not bridges:
        return 0.0
    normalized = text.lower()
    hits = sum(1 for bridge in bridges[:6] if bridge and bridge.lower() in normalized)
    return min(hits * 0.03, 0.12)


def _chunk_signal_bonus(chunk: Chunk) -> float:
    bonus = 0.0
    if chunk.metadata.get("evidence_origin") == "shuiyuan_search_excerpt":
        bonus -= 0.05
    if bool(chunk.metadata.get("topic_is_wiki")):
        bonus += 0.08
    support_count = int(chunk.metadata.get("support_count") or 0)
    bonus += min(support_count, 3) * 0.02
    return bonus


def _expansion_budget(question_shape: str) -> int:
    if question_shape in {"single_fact", "procedure", "time_sensitive_rule"}:
        return 5 if question_shape == "time_sensitive_rule" else 4
    if question_shape in {"enumeration", "comparison", "recommendation"}:
        return 6
    return _BODY_FETCH_CANDIDATE_LIMIT


def _diversify_fetch_candidates(
    candidates: list[CommunitySearchResult],
    *,
    budget: int,
) -> list[CommunitySearchResult]:
    selected: list[CommunitySearchResult] = []
    seen_signatures: list[set[str]] = []
    for item in candidates:
        signature = _support_terms(item.title, item.snippet)
        if any(_signature_overlap(signature, existing) >= 0.7 for existing in seen_signatures):
            continue
        selected.append(item)
        seen_signatures.append(signature)
        if len(selected) >= budget:
            break
    if len(selected) < budget:
        for item in candidates:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= budget:
                break
    return selected


def _signature_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _fallback_structured_evidence() -> dict[str, object]:
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


def _evidence_items(
    evidence_hits: list[RetrievalResult],
    ranked_results: list[CommunitySearchResult],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    seen: set[str] = set()
    for hit in evidence_hits[:_LEDGER_CANDIDATE_LIMIT]:
        evidence_id = hit.chunk.id
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        items.append(
            {
                "evidence_id": evidence_id,
                "title": hit.chunk.metadata.get("topic_title") or hit.chunk.metadata.get("document_title"),
                "url": hit.chunk.metadata.get("post_url") or hit.chunk.metadata.get("topic_url") or hit.chunk.metadata.get("document_uri"),
                "origin": hit.chunk.metadata.get("evidence_origin", ""),
                "text": hit.chunk.text,
                "topic_is_wiki": bool(hit.chunk.metadata.get("topic_is_wiki")),
                "is_solution": bool(hit.chunk.metadata.get("is_solution")),
            }
        )
    for index, result in enumerate(ranked_results[:20], start=1):
        evidence_id = f"search::{index}::{result.url}"
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        items.append(
            {
                "evidence_id": evidence_id,
                "title": result.title,
                "url": result.url,
                "origin": "shuiyuan_search_excerpt",
                "text": result.snippet,
                "topic_is_wiki": result.is_wiki,
                "has_solution": result.has_solution,
                "support_count": result.support_count,
            }
        )
    return items[:_LEDGER_CANDIDATE_LIMIT]


def _fallback_evidence_ledger(evidence_items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "current_answers": [
            {
                "claim": str(item.get("text", "")),
                "evidence_ids": [str(item.get("evidence_id", ""))],
                "confidence": "unverified",
            }
            for item in evidence_items[:3]
        ],
        "stable_support": [],
        "historical_background": [],
        "direct_answers": [
            {
                "claim": str(item.get("text", "")),
                "evidence_ids": [str(item.get("evidence_id", ""))],
                "confidence": "unverified",
            }
            for item in evidence_items[:5]
        ],
        "useful_support": [],
        "background_only": [],
        "remaining_unknowns": ["LLM 事实整理不可用，以下内容仅按检索相关性展示。"],
    }


def _extract_core_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if _is_cjk(raw):
            for segment in _split_cjk_core_segments(raw):
                if len(segment) < 2:
                    continue
                if len(segment) <= 8:
                    terms.add(segment)
                for size in (2, 3, 4):
                    if len(segment) < size:
                        continue
                    terms.update(
                        segment[index : index + size]
                        for index in range(len(segment) - size + 1)
                    )
            continue
        if len(raw) >= 2 and raw not in _QUESTION_STOP_TERMS:
            terms.add(raw)
    return terms


def _split_cjk_core_segments(text: str) -> list[str]:
    segments = [text]
    for stop_term in _QUESTION_STOP_TERMS:
        next_segments: list[str] = []
        for segment in segments:
            next_segments.extend(part for part in segment.split(stop_term) if part)
        segments = next_segments or segments
    return [segment for segment in segments if segment and segment not in _QUESTION_STOP_TERMS]


def _contains_core_term(text: str, core_terms: set[str]) -> bool:
    if not core_terms:
        return bool(text.strip())
    normalized = re.sub(r"\s+", "", text).lower()
    return any(term in normalized for term in core_terms)


def _is_cjk(token: str) -> bool:
    return bool(token) and all("\u4e00" <= char <= "\u9fff" for char in token)


def _character_ngrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}


def _overlap(query: set[str], candidate: set[str]) -> float:
    if not query:
        return 0.0
    return len(query & candidate) / len(query)


def _generate_extractive_answer(results: list[CommunitySearchResult]) -> str:
    if not results:
        return "未在本次 Shuiyuan 社区搜索中找到足够相关的帖子。请调整问题描述后重试。"
    lines = ["根据当前 Shuiyuan 社区帖子，可以先参考这些线索：", ""]
    for index, result in enumerate(results[:3], start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   {result.snippet}")
    lines.extend(["", "参考帖子："])
    for index, result in enumerate(results[:3], start=1):
        lines.append(f"{index}. {result.title} - {result.url}")
    return "\n".join(lines)
