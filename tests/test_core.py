import unittest

import pandas as pd
import torch
from torch import nn

from models.prompt_attention import PromptGuidedLatentAttention
from utils.prototypes import (
    PrototypeSet,
    class_prototypes,
    pairwise_euclidean,
    prompt_tuned_regularization,
)
from utils.sampling import sample_few_shot_task


class PrototypeTests(unittest.TestCase):
    def test_class_prototypes_and_distance(self):
        embeddings = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]])
        labels = torch.tensor([0, 0, 1])
        prototypes = class_prototypes(embeddings, labels, num_classes=2)
        torch.testing.assert_close(
            prototypes, torch.tensor([[1.0, 1.0], [4.0, 0.0]])
        )
        distances = pairwise_euclidean(torch.tensor([[1.0, 1.0]]), prototypes)
        torch.testing.assert_close(distances, torch.tensor([[0.0, 3.1622777]]))

    def test_prompt_regularization_is_finite(self):
        prototype_set = PrototypeSet(
            audio=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            visual=torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            text=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        regularization, metrics = prompt_tuned_regularization(
            torch.tensor([[1.0, 0.0], [0.1, 0.9]]),
            torch.tensor([[0.8, 0.2], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
            prototype_set,
            nn.CrossEntropyLoss(),
            alpha=1.0,
        )
        self.assertTrue(torch.isfinite(regularization))
        self.assertGreaterEqual(metrics["prompt_weight"], 0.0)
        self.assertLessEqual(metrics["prompt_weight"], 1.0)


class LatentAttentionTests(unittest.TestCase):
    def test_shape_preservation(self):
        module = PromptGuidedLatentAttention()
        audio = torch.randn(2, 9, 4)
        visual = torch.randn(2, 13, 4)
        text = torch.randn(2, 5, 4)
        outputs = module(audio, visual, text)
        self.assertEqual(outputs[0].shape, audio.shape)
        self.assertEqual(outputs[1].shape, visual.shape)
        self.assertEqual(outputs[2].shape, text.shape)


class SamplingTests(unittest.TestCase):
    def test_task_labels_are_local_and_balanced(self):
        support_pool = pd.DataFrame(
            {
                "clip_id": [f"clip_{index}" for index in range(8)],
                "label": [10] * 4 + [20] * 4,
                "prompt": ["prompt"] * 8,
            }
        )
        query_pool = support_pool.copy()
        generator = __import__("numpy").random.default_rng(42)
        support, query, class_map = sample_few_shot_task(
            support_pool, query_pool, [10, 20], 2, generator
        )
        self.assertEqual(class_map, {10: 0, 20: 1})
        self.assertEqual(support["label"].value_counts().to_dict(), {0: 2, 1: 2})
        self.assertEqual(set(query["label"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
