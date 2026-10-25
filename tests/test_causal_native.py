import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_native import native_counterfactual, parse_native_params


class NativeDeterministicTest(unittest.TestCase):
    def test_abduction_conditions_before_intervention(self):
        params = {"p(U)": .5, "p(X | U)": [0, 1], "p(Y | X, U)": [[0, 1], [1, 1]]}
        original = native_counterfactual(params, {"X": 1, "Y": 1}, {"X": 0}, "Y")
        self.assertEqual(original, {"value": 1, "outcomes": [1], "compatible_worlds": 1})
        unconditional = native_counterfactual(params, {}, {"X": 0}, "Y")
        self.assertIsNone(unconditional["value"])
        self.assertEqual(unconditional["outcomes"], [0, 1])

    def test_intervened_observation_is_not_reimposed(self):
        result = native_counterfactual({"p(X)": .5, "p(Y | X)": [1, 0]}, {"X": 1}, {"X": 0}, "Y")
        self.assertEqual(result["value"], 1)

    def test_degenerate_root_support_is_preserved(self):
        result = native_counterfactual({"p(X)": 0.0, "p(Y | X)": [0, 1]}, {}, {}, "Y")
        self.assertEqual(result["compatible_worlds"], 1)
        self.assertEqual(result["value"], 0)

    def test_rejects_nondeterministic_children_and_cycles(self):
        for params in ({"p(X)": .5, "p(Y | X)": [.2, .8]},
                       {"p(X | Y)": [0, 1], "p(Y | X)": [0, 1]}):
            with self.assertRaises(ValueError):
                parse_native_params(params)


if __name__ == "__main__":
    unittest.main()
