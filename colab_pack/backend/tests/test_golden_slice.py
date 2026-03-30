import os
import unittest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_WEIGHTS = os.path.join(_BACKEND, "model", "saved", "model.safetensors")


@unittest.skipUnless(os.path.isfile(_MODEL_WEIGHTS), "No saved model weights")
class TestGoldenSlice(unittest.TestCase):
    def test_golden_meets_minimum(self):
        from model.eval_golden import run_golden_eval

        golden = os.path.join(_BACKEND, "data", "golden_eval.json")
        acc, failures, n = run_golden_eval(golden_path=golden, model_dir=None)
        self.assertGreater(
            acc,
            0.55,
            msg=f"Golden accuracy {acc:.3f} on {n} cases; failures={failures}",
        )


if __name__ == "__main__":
    unittest.main()
