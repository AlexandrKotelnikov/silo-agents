from __future__ import annotations

import re
from typing import Any, cast

from .models import RetrievalRecord

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_%-]+")
_SECRET_TOKEN = re.compile(r"\b[A-ZА-ЯЁ]{3,}(?:[_-][A-ZА-ЯЁ0-9]+)*-\d{3,}\b")
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:[,;]|\b(?:as\s+well\s+as|together\s+with|and|plus)\b|"
    r"\b(?:а\s+также|вместе\s+с|и)\b)\s*",
    flags=re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "a", "all", "an", "and", "are", "as", "at", "be", "by", "every", "for",
        "from", "in", "inside", "is", "it", "me", "of", "on", "one", "or", "the",
        "to", "under", "used", "what", "which", "will", "with",
        "administrator", "approval", "assess", "assessment", "authorizes", "bypass",
        "bypassing", "code", "codes", "combine", "command", "commands", "confidential",
        "credentials", "disclose", "document", "documents", "embedded", "execute", "exempt",
        "follow", "ignore", "instruction", "instructions", "list", "note", "policy", "pretend",
        "print", "report", "requested", "reveal", "secret", "secrets", "sensitive", "together",
        "effect", "increase", "limit", "production",
        "в", "все", "вместе", "для", "и", "из", "к", "как", "какая", "какие", "какой",
        "на", "о", "об", "от", "по", "под", "при", "с", "со", "это",
        "администратор", "выполни", "документ", "документа", "документы", "игнорируй",
        "инструкция", "код", "коды", "команда", "комментарий", "объедините", "оцените",
        "перечисли", "политика", "правила", "раскрой", "секретные", "секретный", "таблица",
        "требуется", "выпуск", "выпуска", "годовой", "ограничение", "производство",
        "производства", "рост", "увеличение", "эффект", "эффекта",
    }
)

_RUSSIAN_SUFFIXES = tuple(
    sorted(
        {
            "иями", "ями", "ами", "его", "ого", "ему", "ому", "ыми", "ими", "ией",
            "иям", "иях", "ать", "ять", "ить", "еть", "ую", "юю", "ая", "яя", "ое",
            "ее", "ые", "ие", "ый", "ий", "ой", "ым", "им", "ом", "ем", "ах", "ях",
            "ам", "ям", "ов", "ев", "ия", "ию", "ью", "а", "я", "ы", "и", "у", "ю",
            "е", "о",
        },
        key=len,
        reverse=True,
    )
)

_DEFAULT_ALIASES = {
    "валют": "currency",
    "давлен": "pressure",
    "доход": "margin",
    "маржинальн": "margin",
    "насос": "pump",
    "охлажден": "cooling",
    "подшипник": "bearing",
    "реактор": "reactor",
    "ремонт": "maintenance",
    "риск": "risk",
    "сценарн": "scenario",
    "сырь": "material",
    "энерг": "energy",
}


def normalized_terms(text: str, aliases: dict[str, str] | None = None) -> set[str]:
    """Return conservative bilingual terms for routing, never secret values."""
    active_aliases = {_light_stem(key.casefold()): value.casefold() for key, value in _DEFAULT_ALIASES.items()}
    if aliases:
        active_aliases.update(
            {_light_stem(key.casefold()): value.casefold() for key, value in aliases.items()}
        )
    terms: set[str] = set()
    for raw in _TOKEN_RE.findall(text.casefold()):
        for part in re.split(r"[_%-]+", raw):
            if not part or part in _STOPWORDS or len(part) < 2:
                continue
            normalized = _light_stem(part)
            if normalized and normalized not in _STOPWORDS and len(normalized) >= 2:
                terms.add(normalized)
                alias = active_aliases.get(normalized)
                if alias:
                    terms.add(alias)
    return terms


def split_query_clauses(query: str) -> tuple[str, ...]:
    stripped = query.strip()
    if not stripped:
        return ()
    candidates = (
        part.strip(" \t\n\r.:!?()[]{}") for part in _CLAUSE_SPLIT.split(stripped)
    )
    clauses = tuple(part for part in candidates if part and normalized_terms(part))
    return clauses if len(clauses) > 1 else (stripped,)


def trusted_routing_text(record: RetrievalRecord, routing_terms: set[str] | None = None) -> str:
    shareable_raw: Any = record.metadata.get("shareable", {})
    shareable = cast(dict[str, Any], shareable_raw) if isinstance(shareable_raw, dict) else {}
    restricted = {
        str(value)
        for value in cast(list[object], record.metadata.get("restricted_fields", []))
    }
    safe_keys = " ".join(key for key in shareable if key not in restricted)
    safe_text = _sanitize_text(record.text, _sensitive_values(record))
    declared = " ".join(sorted(routing_terms or set()))
    return " ".join((safe_text, safe_keys, declared))


def routing_score(
    query: str,
    record: RetrievalRecord,
    *,
    routing_terms: set[str] | None = None,
    routing_aliases: dict[str, str] | None = None,
) -> float:
    query_terms = normalized_terms(query, routing_aliases)
    if not query_terms:
        return 0.0
    evidence_terms = normalized_terms(
        trusted_routing_text(record, routing_terms), routing_aliases
    )
    overlap = query_terms & evidence_terms
    return len(overlap) / len(query_terms)


def relevant_records(
    query: str,
    records: list[RetrievalRecord],
    *,
    routing_terms: set[str] | None = None,
    routing_aliases: dict[str, str] | None = None,
) -> list[RetrievalRecord]:
    return [
        record
        for record in records
        if routing_score(
            query,
            record,
            routing_terms=routing_terms,
            routing_aliases=routing_aliases,
        )
        > 0
    ]


def safe_shareable(record: RetrievalRecord) -> dict[str, Any]:
    shareable_raw: Any = record.metadata.get("shareable", {})
    shareable = cast(dict[str, Any], shareable_raw) if isinstance(shareable_raw, dict) else {}
    restricted = {
        str(value)
        for value in cast(list[object], record.metadata.get("restricted_fields", []))
    }
    return {key: value for key, value in shareable.items() if key not in restricted}


def safe_llm_record(record: RetrievalRecord) -> RetrievalRecord:
    summary = _safe_summary(record)
    return RetrievalRecord(
        record_id=record.record_id,
        domain=record.domain,
        text=summary,
        classification=record.classification,
        metadata={"summary": summary, "shareable": safe_shareable(record)},
    )


def _sensitive_values(record: RetrievalRecord) -> set[str]:
    shareable_raw: Any = record.metadata.get("shareable", {})
    shareable = cast(dict[str, Any], shareable_raw) if isinstance(shareable_raw, dict) else {}
    restricted = {
        str(value)
        for value in cast(list[object], record.metadata.get("restricted_fields", []))
    }
    return {str(shareable[key]) for key in restricted if key in shareable}


def _safe_summary(record: RetrievalRecord) -> str:
    return _sanitize_text(str(record.metadata.get("summary", record.text)), _sensitive_values(record))


def _sanitize_text(text: str, sensitive_values: set[str]) -> str:
    result = text
    for sensitive in sorted(sensitive_values, key=len, reverse=True):
        if sensitive:
            result = result.replace(sensitive, "[REDACTED]")
    return _SECRET_TOKEN.sub("[REDACTED]", result)


def _light_stem(token: str) -> str:
    if not re.search(r"[а-яё]", token):
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token
    for suffix in _RUSSIAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token
