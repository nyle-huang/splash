"""Tests for prompt rubric checks and output scorer interfaces."""

from __future__ import annotations

import unittest

from product_campaign_pipeline.composer import FluxPrompt
from product_campaign_pipeline.evaluation import (
    CompositeOutputScorer,
    GeneratedOutputCandidate,
    OutputCriterionScore,
    OutputScore,
    PromptRubric,
)


class _StubScorer:
    name = "stub"

    def score(self, candidate: GeneratedOutputCandidate) -> OutputScore:
        del candidate
        return OutputScore(
            evaluator_name=self.name,
            criteria=(OutputCriterionScore(name="aesthetic_performance", score=0.9),),
        )


class EvaluationTests(unittest.TestCase):
    def test_prompt_rubric_passes_structured_campaign_prompt(self) -> None:
        prompt = FluxPrompt(
            subject="The exact navy bottle with logo and cap preserved from the reference image.",
            action="Stage the bottle as the hero in a campaign scene with a redesigned background.",
            style="High-end commercial photography with realistic reflections and premium lighting.",
            context="Use a believable kitchen environment with depth and supporting props instead of a catalog sweep.",
            preservation_constraints=(
                "Preserve the exact product identity and branding.",
                "Keep materials, silhouette, and colors unchanged.",
            ),
            negative_constraints=("Do not generate a plain white catalog background.",),
        )
        result = PromptRubric().evaluate(prompt)

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.average_score, 0.8)

    def test_output_scorer_interface_composes_multiple_scorers(self) -> None:
        candidate = GeneratedOutputCandidate(image_path="/tmp/fake.png")
        scores = CompositeOutputScorer((_StubScorer(), _StubScorer())).score(candidate)

        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0].average_score, 0.9)


if __name__ == "__main__":
    unittest.main()
