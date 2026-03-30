import unittest
from types import SimpleNamespace

import torch

from model.predict import predict


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, padding=True, max_length=128):
        _ = (text, return_tensors, truncation, padding, max_length)
        return {
            "input_ids": torch.tensor([[101, 2023, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, stressed_logit, not_stressed_logit):
        self.stressed_logit = stressed_logit
        self.not_stressed_logit = not_stressed_logit

    def __call__(self, **kwargs):
        _ = kwargs
        return SimpleNamespace(
            logits=torch.tensor([[self.not_stressed_logit, self.stressed_logit]], dtype=torch.float32)
        )


class TestPredictSmoke(unittest.TestCase):
    def test_predict_non_uncertain_label(self):
        model = _FakeModel(stressed_logit=2.5, not_stressed_logit=0.1)
        result = predict("I am overwhelmed", model, _FakeTokenizer(), torch.device("cpu"))
        self.assertEqual(result["label"], "Stressed")
        self.assertFalse(result["is_uncertain"])
        self.assertGreater(result["probability"], 0.5)

    def test_predict_uncertain_label(self):
        model = _FakeModel(stressed_logit=0.02, not_stressed_logit=0.0)
        result = predict("I feel okay maybe", model, _FakeTokenizer(), torch.device("cpu"))
        self.assertEqual(result["label"], "Uncertain")
        self.assertTrue(result["is_uncertain"])


if __name__ == "__main__":
    unittest.main()
