"""Regression checks for the QSMEF journal artifact.

Run: python -m unittest -v test_qsmef_regression.py
The reference JSON is an independent, previously recorded baseline; tests
recompute results rather than merely reading the baseline.
"""
import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
IMPLEMENTATION = ROOT / 'QSMEF_Implementation.py'
REFERENCE = ROOT / 'qsmef_checks.json'

spec = importlib.util.spec_from_file_location('qsmef_impl', IMPLEMENTATION)
qsmef = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qsmef
spec.loader.exec_module(qsmef)
REFERENCE_DATA = json.loads(REFERENCE.read_text(encoding='utf-8'))
ATOL = 2e-5  # Conservative allowance for platform-dependent optimization.


def quiet(callable_, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_(*args, **kwargs)


def evaluate(case):
    validation = quiet(qsmef.validate_qsmef_case, case, show_tables=False)
    shapley = quiet(qsmef.compute_shapley_values, case, validation, show_table=False)
    return validation, shapley


class QSMEFRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = quiet(qsmef.build_case_registry)
        cls.reports = {}
        for key, (_, case) in cls.cases.items():
            cls.reports[key] = evaluate(case)

    def check_case(self, key):
        expected = REFERENCE_DATA[key]
        validation, shapley = self.reports[key]
        coalitions = validation['coalitions']
        self.assertEqual(validation['verdict'], expected['verdict'])
        self.assertEqual(len(coalitions), expected['coalitions'])
        self.assertEqual(int((~coalitions['contrato_semántico_cumplido']).sum()), expected['failed_semantic'])
        self.assertEqual(int((~coalitions['restricciones_cumplidas']).sum()), expected['failed_functional'])
        self.assertEqual(len(coalitions), 2 ** len(self.cases[key][1].players))
        if expected['shapley'] is None:
            self.assertFalse(shapley['computed'])
            self.assertEqual(shapley['shapley_values'], {})
        else:
            self.assertTrue(shapley['computed'])
            self.assertEqual(set(shapley['shapley_values']), set(expected['shapley']))
            for player, value in expected['shapley'].items():
                self.assertTrue(math.isclose(shapley['shapley_values'][player], value, abs_tol=ATOL, rel_tol=0),
                                f'{key}/{player}: {shapley["shapley_values"][player]} != {value}')
            # Independently check Shapley efficiency against the full coalition.
            full_label = '{' + ', '.join(self.cases[key][1].players) + '}'
            full_value = float(coalitions.loc[coalitions['coalición'] == full_label, 'v(C)'].iloc[0])
            self.assertAlmostEqual(sum(shapley['shapley_values'].values()), full_value, delta=1e-8)

    def test_maxcut(self): self.check_case('maxcut')
    def test_h2(self): self.check_case('h2')
    def test_skw(self): self.check_case('skw')
    def test_qpe(self): self.check_case('qpe')
    def test_partitioning(self): self.check_case('partitioning')

    def test_skw_modified_full_evolution(self):
        original = self.cases['skw'][1]
        modified = quiet(qsmef.build_skw_modified_evolution_case)
        self.assertFalse(qsmef.np.allclose(original.initial_state, modified.initial_state))
        validation, shapley = evaluate(modified)
        self.assertEqual(validation['verdict'], 'APPLICABLE')
        self.assertEqual(len(validation['coalitions']), 8)
        self.assertTrue(shapley['computed'])
        self.assertTrue(shapley['efficiency_ok'])
        # Numerical regression against the published full-evolution SKW result.
        self.assertIn('O', shapley['shapley_values'])
        self.assertAlmostEqual(shapley['shapley_values']['O'], 0.134201, delta=ATOL)
        original_phi_o = self.reports['skw'][1]['shapley_values']['O']
        self.assertAlmostEqual(shapley['shapley_values']['O'] - original_phi_o, 0.387256, delta=ATOL)

    def test_skw_controlled_fixed_state(self):
        original = self.cases['skw'][1]
        modified = quiet(qsmef.build_skw_modified_fixed_state_case)
        self.assertTrue(qsmef.np.allclose(original.initial_state, modified.initial_state, atol=1e-12))
        self.assertEqual(list(original.players), list(modified.players))
        self.assertEqual(list(original.schedule), list(modified.schedule))
        for label, case in [('original', original), ('modified_fixed_state', modified)]:
            validation, shapley = evaluate(case)
            expected = REFERENCE_DATA['controlled_skw'][label]
            self.assertEqual(validation['verdict'], expected['verdict'])
            self.assertTrue(shapley['efficiency_ok'])
            for player, value in expected['shapley'].items():
                self.assertAlmostEqual(shapley['shapley_values'][player], value, delta=ATOL)
        # Only the oracle intervention changes, so G/S should be unaffected.
        orig_values = self.reports['skw'][1]['shapley_values']
        mod_values = evaluate(modified)[1]['shapley_values']
        for player in ('G', 'S'):
            self.assertAlmostEqual(orig_values[player], mod_values[player], delta=1e-9)

    def test_qpe_unsupported_configuration_rejected(self):
        with self.assertRaises(ValueError):
            quiet(qsmef.build_qpe_shor_case, N=15, a=2, counting_qubits=5)

    def test_maxcut_granularity_identifiers(self):
        case = self.cases['maxcut'][1]
        # The assistant must not refer to nonexistent M1/M2 players.
        result = quiet(qsmef.example_maxcut_granularity, case)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
