import unittest

from fastapi.testclient import TestClient

import app as backend_app


class TestApiSmoke(unittest.TestCase):
    def setUp(self):
        self._orig_startup = list(backend_app.app.router.on_startup)
        backend_app.app.router.on_startup = []
        self.client = TestClient(backend_app.app)
        self._orig_predict = backend_app.predict
        self._orig_explain = backend_app.explain
        backend_app._model = object()
        backend_app._tokenizer = object()
        backend_app._device = "cpu"

    def tearDown(self):
        backend_app.predict = self._orig_predict
        backend_app.explain = self._orig_explain
        backend_app.app.router.on_startup = self._orig_startup

    def test_predict_contract_normal(self):
        def fake_predict(_text, _model, _tokenizer, _device):
            return {
                "label": "Stressed",
                "probability": 0.9012,
                "is_uncertain": False,
                "recommended_action": "Prediction confidence is sufficient for this sample.",
            }

        def fake_explain(_text, _model, _tokenizer, _device, prediction_context=None):
            _ = prediction_context
            return {
                "shap": {"overwhelmed": 0.13},
                "lime": {"overwhelmed": 0.22},
                "attention": {"overwhelmed": 1.0},
                "agreement": {"score": 0.66, "consensus_words": ["overwhelmed"]},
                "emotion_signals": {
                    "dominant_emotion": "anxiety",
                    "scores": {"anxiety": 1.0, "burnout": 0.0, "pressure": 0.0, "distress": 0.0},
                },
            }

        backend_app.predict = fake_predict
        backend_app.explain = fake_explain

        response = self.client.post("/predict", json={"text": "I feel overwhelmed with work."})
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("label", body)
        self.assertIn("probability", body)
        self.assertIn("is_uncertain", body)
        self.assertIn("recommended_action", body)
        self.assertIn("explanations", body)

        self.assertIn("agreement", body["explanations"])
        self.assertIn("emotion_signals", body["explanations"])

    def test_predict_contract_uncertain_and_partial_explanations(self):
        def fake_predict(_text, _model, _tokenizer, _device):
            return {
                "label": "Uncertain",
                "probability": 0.5001,
                "is_uncertain": True,
                "recommended_action": "Provide more context for reliable estimate.",
            }

        def fake_explain(_text, _model, _tokenizer, _device, prediction_context=None):
            _ = prediction_context
            return {
                "shap": {},
                "lime": {},
                "attention": {},
                "agreement": {"score": 0.0, "consensus_words": []},
                "emotion_signals": {"dominant_emotion": "none", "scores": {}},
            }

        backend_app.predict = fake_predict
        backend_app.explain = fake_explain

        response = self.client.post("/predict", json={"text": "Not sure what I feel."})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["label"], "Uncertain")
        self.assertTrue(body["is_uncertain"])
        self.assertEqual(body["explanations"]["agreement"]["score"], 0.0)

    def test_predict_rejects_empty_text(self):
        response = self.client.post("/predict", json={"text": "   "})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
