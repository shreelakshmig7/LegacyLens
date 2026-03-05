"""
__init__.py
-----------
LegacyLens — RAG System for Legacy Enterprise Codebases — Features package initializer
---------------------------------------------------------------------------------------
The features package implements higher-level code understanding capabilities built on
top of the retrieval layer: code explanation, dependency mapping, business logic
extraction, and documentation generation.

Provides detect_feature_type() which classifies an incoming query and routes it to
the appropriate feature module.  Priority order (highest first):
    dependency → explain → business_logic → doc_generate → general

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import logging
import re
from typing import Optional

from legacylens.config.constants import (
    BUSINESS_LOGIC_KEYWORDS,
    DEPENDENCY_QUERY_KEYWORDS,
    DOC_GENERATE_KEYWORDS,
    EXPLAIN_QUERY_KEYWORDS,
    FEATURE_TYPE_BUSINESS_LOGIC,
    FEATURE_TYPE_DEPENDENCY,
    FEATURE_TYPE_DOC_GENERATE,
    FEATURE_TYPE_EXPLAIN,
    FEATURE_TYPE_GENERAL,
)

logger = logging.getLogger(__name__)

# Ordered list: checked first → highest priority
_FEATURE_KEYWORD_MAP = [
    (FEATURE_TYPE_DEPENDENCY, DEPENDENCY_QUERY_KEYWORDS),
    (FEATURE_TYPE_EXPLAIN, EXPLAIN_QUERY_KEYWORDS),
    (FEATURE_TYPE_BUSINESS_LOGIC, BUSINESS_LOGIC_KEYWORDS),
    (FEATURE_TYPE_DOC_GENERATE, DOC_GENERATE_KEYWORDS),
]


def detect_feature_type(query: Optional[str]) -> str:
    """
    Classify a query into a feature type for routing to the correct feature module.

    Checks the query against keyword lists in priority order (dependency first,
    then explain, business_logic, doc_generate). Keywords may contain regex
    patterns (e.g. ``what does .* call``).  All matching is case-insensitive.

    Args:
        query: The user's natural language query string. May be None or empty.

    Returns:
        str: One of the FEATURE_TYPE_* constants from config.constants —
             "dependency", "explain", "business_logic", "doc_generate", or "general".
    """
    if not query or not isinstance(query, str) or not query.strip():
        return FEATURE_TYPE_GENERAL

    lower = query.lower().strip()

    for feature_type, keywords in _FEATURE_KEYWORD_MAP:
        for keyword in keywords:
            try:
                if re.search(keyword.lower(), lower):
                    logger.debug(
                        "Query routed to '%s' (matched keyword: '%s')",
                        feature_type,
                        keyword,
                    )
                    return feature_type
            except re.error:
                if keyword.lower() in lower:
                    return feature_type

    return FEATURE_TYPE_GENERAL
