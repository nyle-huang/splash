"""Tests for the CTR-aware retrieval planner."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from product_campaign_pipeline.planner import (
    CTRAwareRetrievalPlanner,
    CreativeRankingItem,
    PlannerInput,
    load_creative_ranking_manifest,
)


class PlannerTests(unittest.TestCase):
    def test_retrieval_prefers_similarity_when_ctr_is_comparable(self) -> None:
        items = [
            CreativeRankingItem(
                item_id="item-a",
                image_name="a.png",
                ds=0,
                page_views=200,
                clicks=80,
                style_hints=("sunlit editorial lighting",),
                context_hints=("outdoor terrace brunch",),
                embedding=(1.0, 0.0),
            ),
            CreativeRankingItem(
                item_id="item-b",
                image_name="b.png",
                ds=0,
                page_views=200,
                clicks=82,
                style_hints=("flat studio lighting",),
                context_hints=("blank studio sweep",),
                embedding=(0.0, 1.0),
            ),
        ]

        plan = CTRAwareRetrievalPlanner(items).plan(
            PlannerInput(product_name="Travel Mug", query_embedding=(0.98, 0.02), top_k=1)
        )

        self.assertEqual(plan.selected[0].item.image_name, "a.png")
        self.assertIn("sunlit editorial lighting", plan.aggregated_style_hints)
        self.assertIn("outdoor terrace brunch", plan.aggregated_context_hints)

    def test_manifest_loader_applies_annotations_and_embeddings(self) -> None:
        manifest_body = textwrap.dedent(
            """\
            item-1\timage-1.png\t0\t10\t2
            item-2\timage-2.png\t1\t20\t5
            """
        )
        annotations = {
            "image-2.png": {
                "style_hints": ["glass reflections", "clean luxury still life"],
                "context_hints": ["stone vanity"],
            }
        }
        embeddings = {"item-2::image-2.png": [0.25, 0.75]}

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.tsv"
            manifest_path.write_text(manifest_body, encoding="utf-8")
            items = load_creative_ranking_manifest(
                manifest_path,
                annotations=annotations,
                embeddings=embeddings,
            )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[1].style_hints, ("glass reflections", "clean luxury still life"))
        self.assertEqual(items[1].embedding, (0.25, 0.75))


if __name__ == "__main__":
    unittest.main()
