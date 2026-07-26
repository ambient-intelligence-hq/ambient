"""Token-usage cost estimation from the OpenRouter-style model catalog.

`settings.model_definitions_file` (artifacts/models.json) carries per-token
prices as strings (USD/token) under each model's `pricing` block:

    "pricing": {"prompt": "0.00001", "completion": "0.00005",
                "input_cache_read": "...", "input_cache_write": "..."}

`estimate_cost(model, usage)` maps a usage dict onto those prices. Matching is
lenient (exact id, then suffix/substring) so `google/gemini-3.1-pro-preview`
still resolves when the catalog id is `google/gemini-3.1-pro`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from ambient.config import settings

import logging

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict]:
    try:
        with open(settings.model_definitions_file, "r") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        log.warning("Could not load model pricing catalog: %s", exc)
        return {}
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    out: dict[str, dict] = {}
    for m in data or []:
        mid = m.get("id")
        if mid and isinstance(m.get("pricing"), dict):
            out[mid] = m["pricing"]
    return out


def _find_pricing(model: str) -> Optional[dict]:
    catalog = _catalog()
    if not model or not catalog:
        return None
    if model in catalog:
        return catalog[model]
    # Suffix match on the model slug (drop any provider prefix mismatch), then a
    # loose substring match so preview/date-suffixed ids still resolve.
    slug = model.split("/")[-1]
    for mid, pricing in catalog.items():
        if mid.split("/")[-1] == slug:
            return pricing
    for mid, pricing in catalog.items():
        if slug in mid or mid.split("/")[-1] in model:
            return pricing
    return None


def _price(pricing: dict, key: str) -> float:
    try:
        return float(pricing.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def estimate_cost(model: str, usage: dict) -> float:
    """USD cost for a single usage dict, or 0.0 when the model isn't priced.

    `usage` keys: input_tokens, output_tokens, cache_read_input_tokens,
    cache_creation_input_tokens (any subset)."""
    pricing = _find_pricing(model)
    if not pricing:
        return 0.0
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    # Cache-read tokens are billed at the cheaper cache-read rate, so bill only the
    # non-cached input at the prompt rate.
    billable_input = max(input_tokens - cache_read, 0)
    cost = (
        billable_input * _price(pricing, "prompt")
        + output_tokens * _price(pricing, "completion")
        + cache_read * _price(pricing, "input_cache_read")
        + cache_write * _price(pricing, "input_cache_write")
    )
    return round(cost, 6)
