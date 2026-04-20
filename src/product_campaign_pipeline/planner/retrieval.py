"""CTR-aware retrieval planner built around CreativeRanking-style item statistics."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Vector = Sequence[float]


def _coerce_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        key = candidate.lower()
        if not candidate or key in seen:
            continue
        normalized.append(candidate)
        seen.add(key)
    return tuple(normalized)


def _coerce_embedding(values: Sequence[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    embedding = tuple(float(value) for value in values)
    return embedding or None


def _normalize_score(value: float, minimum: float, maximum: float) -> float:
    if math.isclose(minimum, maximum):
        return 1.0
    return (value - minimum) / (maximum - minimum)


def _cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass(frozen=True)
class CreativeRankingItem:
    """CreativeRanking item plus optional campaign metadata for prompt retrieval."""

    item_id: str
    image_name: str
    ds: int | str
    page_views: int
    clicks: int
    subject_hints: tuple[str, ...] = ()
    action_hints: tuple[str, ...] = ()
    style_hints: tuple[str, ...] = ()
    context_hints: tuple[str, ...] = ()
    preservation_hints: tuple[str, ...] = ()
    negative_hints: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_hints", _coerce_tuple(self.subject_hints))
        object.__setattr__(self, "action_hints", _coerce_tuple(self.action_hints))
        object.__setattr__(self, "style_hints", _coerce_tuple(self.style_hints))
        object.__setattr__(self, "context_hints", _coerce_tuple(self.context_hints))
        object.__setattr__(self, "preservation_hints", _coerce_tuple(self.preservation_hints))
        object.__setattr__(self, "negative_hints", _coerce_tuple(self.negative_hints))
        object.__setattr__(self, "embedding", _coerce_embedding(self.embedding))
        object.__setattr__(self, "page_views", max(int(self.page_views), 0))
        object.__setattr__(self, "clicks", max(int(self.clicks), 0))

    @property
    def ctr(self) -> float:
        if self.page_views <= 0:
            return 0.0
        return self.clicks / self.page_views


@dataclass(frozen=True)
class PlannerInput:
    """Input product description used to retrieve high-performing creative priors."""

    product_name: str
    category: str | None = None
    subject_description: str | None = None
    product_attributes: tuple[str, ...] = ()
    preservation_constraints: tuple[str, ...] = ()
    query_embedding: tuple[float, ...] | None = None
    top_k: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_attributes", _coerce_tuple(self.product_attributes))
        object.__setattr__(
            self,
            "preservation_constraints",
            _coerce_tuple(self.preservation_constraints),
        )
        object.__setattr__(self, "query_embedding", _coerce_embedding(self.query_embedding))
        object.__setattr__(self, "top_k", max(int(self.top_k), 1))


@dataclass(frozen=True)
class RetrievedCreative:
    """Single retrieved example with score breakdown for debugging and inspection."""

    item: CreativeRankingItem
    score: float
    smoothed_ctr: float
    similarity: float
    popularity: float


@dataclass(frozen=True)
class RetrievalPlan:
    """Aggregated planner output that conditions the business-prior composer."""

    planner_name: str
    query: PlannerInput
    selected: tuple[RetrievedCreative, ...]
    aggregated_subject_hints: tuple[str, ...]
    aggregated_action_hints: tuple[str, ...]
    aggregated_style_hints: tuple[str, ...]
    aggregated_context_hints: tuple[str, ...]
    aggregated_preservation_hints: tuple[str, ...]
    aggregated_negative_hints: tuple[str, ...]

    def as_prompt_context(self) -> dict[str, list[str]]:
        return {
            "subject_hints": list(self.aggregated_subject_hints),
            "action_hints": list(self.aggregated_action_hints),
            "style_hints": list(self.aggregated_style_hints),
            "context_hints": list(self.aggregated_context_hints),
            "preservation_hints": list(self.aggregated_preservation_hints),
            "negative_hints": list(self.aggregated_negative_hints),
        }


class CTRAwareRetrievalPlanner:
    """Retrieval planner that combines smoothed CTR, popularity, and optional embeddings."""

    def __init__(
        self,
        items: Sequence[CreativeRankingItem],
        *,
        ctr_weight: float = 0.7,
        similarity_weight: float = 0.25,
        popularity_weight: float = 0.05,
        ctr_prior_strength: float = 8.0,
        hint_limit: int = 3,
    ) -> None:
        self._items = tuple(items)
        if not self._items:
            raise ValueError("CTRAwareRetrievalPlanner requires at least one item")
        self._ctr_weight = float(ctr_weight)
        self._similarity_weight = float(similarity_weight)
        self._popularity_weight = float(popularity_weight)
        self._ctr_prior_strength = max(float(ctr_prior_strength), 0.0)
        self._hint_limit = max(int(hint_limit), 1)
        ctr_values = [item.ctr for item in self._items]
        self._ctr_prior_mean = sum(ctr_values) / len(ctr_values)
        popularity_values = [math.log1p(item.page_views) for item in self._items]
        self._popularity_min = min(popularity_values)
        self._popularity_max = max(popularity_values)

    @property
    def items(self) -> tuple[CreativeRankingItem, ...]:
        return self._items

    def plan(self, query: PlannerInput) -> RetrievalPlan:
        scored: list[RetrievedCreative] = []
        use_similarity = query.query_embedding is not None

        for item in self._items:
            smoothed_ctr = self._smoothed_ctr(item)
            popularity = _normalize_score(
                math.log1p(item.page_views),
                self._popularity_min,
                self._popularity_max,
            )
            similarity = 0.0
            if use_similarity and item.embedding is not None:
                similarity = max(
                    _cosine_similarity(query.query_embedding or (), item.embedding),
                    0.0,
                )

            weight_sum = self._ctr_weight + self._popularity_weight
            score = self._ctr_weight * smoothed_ctr + self._popularity_weight * popularity
            if use_similarity:
                weight_sum += self._similarity_weight
                score += self._similarity_weight * similarity
            scored.append(
                RetrievedCreative(
                    item=item,
                    score=score / weight_sum,
                    smoothed_ctr=smoothed_ctr,
                    similarity=similarity,
                    popularity=popularity,
                )
            )

        selected = tuple(
            sorted(
                scored,
                key=lambda candidate: (
                    candidate.score,
                    candidate.similarity,
                    candidate.smoothed_ctr,
                    candidate.popularity,
                ),
                reverse=True,
            )[: query.top_k]
        )

        return RetrievalPlan(
            planner_name=self.__class__.__name__,
            query=query,
            selected=selected,
            aggregated_subject_hints=self._aggregate_hints(selected, "subject_hints"),
            aggregated_action_hints=self._aggregate_hints(selected, "action_hints"),
            aggregated_style_hints=self._aggregate_hints(selected, "style_hints"),
            aggregated_context_hints=self._aggregate_hints(selected, "context_hints"),
            aggregated_preservation_hints=self._aggregate_hints(
                selected,
                "preservation_hints",
            ),
            aggregated_negative_hints=self._aggregate_hints(selected, "negative_hints"),
        )

    def _aggregate_hints(
        self,
        selected: Sequence[RetrievedCreative],
        attribute_name: str,
    ) -> tuple[str, ...]:
        weighted: dict[str, float] = {}
        labels: dict[str, str] = {}

        for candidate in selected:
            hints = getattr(candidate.item, attribute_name)
            for hint in hints:
                key = hint.strip().lower()
                if not key:
                    continue
                weighted[key] = weighted.get(key, 0.0) + max(candidate.score, 0.01)
                labels.setdefault(key, hint.strip())

        ordered = sorted(
            weighted.items(),
            key=lambda entry: (-entry[1], labels[entry[0]].lower()),
        )
        return tuple(labels[key] for key, _ in ordered[: self._hint_limit])

    def _smoothed_ctr(self, item: CreativeRankingItem) -> float:
        numerator = item.clicks + self._ctr_prior_mean * self._ctr_prior_strength
        denominator = item.page_views + self._ctr_prior_strength
        if math.isclose(denominator, 0.0):
            return self._ctr_prior_mean
        return numerator / denominator


def load_creative_ranking_manifest(
    manifest_path: str | Path,
    *,
    annotations: Mapping[str, Mapping[str, Any]] | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
) -> list[CreativeRankingItem]:
    """Load a CreativeRanking-style TSV manifest.

    The manifest is expected to contain the schema documented in the repo:
    ``item_id, image_name, ds, pv, clk`` separated by tabs.

    ``annotations`` and ``embeddings`` may be keyed by ``image_name``,
    ``item_id``, or ``item_id::image_name``.
    """

    annotations = annotations or {}
    embeddings = embeddings or {}
    items: list[CreativeRankingItem] = []

    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(
                    f"Expected 5 tab-separated fields per row in {manifest_path}, got {len(row)}"
                )
            item_id, image_name, ds, page_views, clicks = row
            key_variants = (f"{item_id}::{image_name}", image_name, item_id)
            annotation = _first_mapping(annotations, key_variants)
            embedding = _first_value(embeddings, key_variants)
            items.append(
                CreativeRankingItem(
                    item_id=item_id,
                    image_name=image_name,
                    ds=_parse_int(ds),
                    page_views=_parse_int(page_views),
                    clicks=_parse_int(clicks),
                    subject_hints=annotation.get("subject_hints", ()),
                    action_hints=annotation.get("action_hints", ()),
                    style_hints=annotation.get("style_hints", ()),
                    context_hints=annotation.get("context_hints", ()),
                    preservation_hints=annotation.get("preservation_hints", ()),
                    negative_hints=annotation.get("negative_hints", ()),
                    embedding=embedding,
                    metadata={
                        "manifest_path": str(manifest_path),
                        **annotation.get("metadata", {}),
                    },
                )
            )
    return items


def _first_mapping(
    mapping: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
) -> Mapping[str, Any]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return {}


def _first_value(
    mapping: Mapping[str, Sequence[float]],
    keys: Sequence[str],
) -> Sequence[float] | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_int(value: str) -> int | str:
    candidate = value.strip()
    if candidate.isdigit() or (candidate.startswith("-") and candidate[1:].isdigit()):
        return int(candidate)
    return candidate
