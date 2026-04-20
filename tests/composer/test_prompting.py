"""Tests for baseline and business-prior FLUX prompt composition."""

from __future__ import annotations

import unittest

from product_campaign_pipeline.composer import BaselineComposer, BusinessPriorComposer, ProductBrief
from product_campaign_pipeline.planner import (
    CTRAwareRetrievalPlanner,
    CreativeRankingItem,
    PlannerInput,
)


class ComposerTests(unittest.TestCase):
    def test_baseline_prompt_uses_ordered_flux_sections(self) -> None:
        prompt = BaselineComposer().compose(
            ProductBrief(
                product_name="OceanMist Bottle",
                category="drinkware",
                product_description="a matte navy stainless steel bottle",
                key_attributes=("matte navy finish", "etched logo", "steel cap"),
            )
        )

        self.assertEqual(
            tuple(prompt.as_dict().keys())[:4],
            ("subject", "action", "style", "context"),
        )
        self.assertIn("exact", prompt.subject.lower())
        self.assertTrue(prompt.preservation_constraints)

    def test_business_prior_prompt_injects_retrieved_creative_hints(self) -> None:
        planner = CTRAwareRetrievalPlanner(
            [
                CreativeRankingItem(
                    item_id="item-1",
                    image_name="hero.png",
                    ds=0,
                    page_views=100,
                    clicks=35,
                    action_hints=("pouring water into a glass",),
                    style_hints=("warm sunrise reflections",),
                    context_hints=("stone counter beside citrus garnish",),
                    preservation_hints=("Keep the cap threading identical",),
                    embedding=(1.0, 0.0),
                )
            ]
        )
        plan = planner.plan(PlannerInput(product_name="Bottle", query_embedding=(1.0, 0.0)))

        prompt = BusinessPriorComposer().compose(
            ProductBrief(product_name="OceanMist Bottle", category="drinkware"),
            plan,
        )

        self.assertIn("pouring water into a glass", prompt.action)
        self.assertIn("warm sunrise reflections", prompt.style)
        self.assertIn("stone counter beside citrus garnish", prompt.context)
        self.assertIn("Keep the cap threading identical", prompt.preservation_constraints)


if __name__ == "__main__":
    unittest.main()
