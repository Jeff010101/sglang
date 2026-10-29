"""Exercise source-owned CPU topology helpers without a server or GPU stack."""

import ast
import logging
import unittest
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TopologyHarness:
    """Load the real helper definitions with only their OS boundaries replaced."""

    CSV = "# CPU,Core,Socket,Node\n0,0,0,0\n1,1,0,0\n2,0,0,0\n4,0,1,1\n5,1,1,1\n"

    def __init__(self, enabled=True):
        self.env = (
            {"SGLANG_SIMULATOR_BOOTSTRAP": "1", "SGLANG_USE_CPU_ENGINE": "1"}
            if enabled
            else {}
        )
        self.pid = 100
        self.online = "0-5\n"
        self.csv = self.CSV
        self.affinity = [0, 1, 2, 4, 5]
        self.read_calls = 0
        self.affinity_calls = 0
        self.read_error = None
        self.online_error = None

        def read_csv(argv, *, text):
            assert argv == ["lscpu", "-p=CPU,Core,Socket,Node"] and text is True
            self.read_calls += 1
            if self.read_error is not None:
                raise self.read_error
            return self.csv

        def read_online():
            if self.online_error is not None:
                raise self.online_error
            return self.online

        def read_affinity():
            self.affinity_calls += 1
            return self.affinity[:]

        def path(name):
            assert name == "/sys/devices/system/cpu/online"
            return SimpleNamespace(read_text=read_online)

        self.scope = {
            "defaultdict": defaultdict,
            "lru_cache": lru_cache,
            "logger": logging.getLogger(__name__),
            "subprocess": SimpleNamespace(check_output=read_csv),
            "psutil": SimpleNamespace(
                Process=lambda: SimpleNamespace(cpu_affinity=read_affinity)
            ),
            "os": SimpleNamespace(getenv=self.env.get, getpid=lambda: self.pid),
            "Path": path,
        }
        source = (
            Path(__file__).resolve().parents[4] / "python/sglang/srt/utils/common.py"
        )
        names = {
            "parse_lscpu_topology",
            "_ORIGINAL_PARSE_LSCPU_TOPOLOGY",
            "_ORIGINAL_LSCPU_CHECK_OUTPUT",
            "_simulator_cpu_topology",
            "_physical_cpu_topology",
            "get_physical_cpus_by_numa",
            "get_cpu_ids_by_node",
        }
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            ):
                selected.append(node)
        assert len(selected) == len(names)
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
            self.scope,
        )

    def physical(self):
        return self.scope["get_physical_cpus_by_numa"]()

    def ids(self):
        return self.scope["get_cpu_ids_by_node"]()


class TestSimulatorCpuTopology(CustomTestCase):
    def test_standard_calls_reuse_topology_with_fresh_public_results(self):
        harness = TopologyHarness()
        first = harness.physical()
        first[0].add(999)
        first[1].clear()
        self.assertEqual(harness.physical(), {0: {0, 1}, 1: {4, 5}})
        self.assertEqual(harness.ids(), ["0,1", "4,5"])
        self.assertEqual(harness.read_calls, 1)
        self.assertEqual(harness.affinity_calls, 3)

    def test_both_exact_opt_in_flags_are_required(self):
        cases = (
            {},
            {"SGLANG_SIMULATOR_BOOTSTRAP": "1"},
            {"SGLANG_USE_CPU_ENGINE": "1"},
            {"SGLANG_SIMULATOR_BOOTSTRAP": "true", "SGLANG_USE_CPU_ENGINE": "1"},
        )
        for env in cases:
            with self.subTest(env=env):
                harness = TopologyHarness(enabled=False)
                harness.env.update(env)
                harness.physical()
                harness.physical()
                self.assertEqual(harness.read_calls, 2)

    def test_affinity_is_live_and_empty_numa_output_is_unchanged(self):
        harness = TopologyHarness()
        harness.physical()
        harness.affinity = [1]
        self.assertEqual(harness.physical(), {0: {1}, 1: set()})
        # Keep this branch's original get_cpu_ids_by_node empty-node behavior.
        self.assertEqual(harness.ids(), ["1", ""])
        self.assertEqual(harness.read_calls, 1)

    def test_pid_and_online_cpu_changes_refresh_topology(self):
        harness = TopologyHarness()
        harness.physical()
        harness.pid += 1
        harness.csv += "6,2,1,1\n"
        harness.affinity.append(6)
        self.assertEqual(harness.physical()[1], {4, 5, 6})
        harness.online = "0-7\n"
        harness.csv += "7,3,1,1\n"
        harness.affinity.append(7)
        self.assertEqual(harness.physical()[1], {4, 5, 6, 7})
        self.assertEqual(harness.read_calls, 3)
        self.assertEqual(
            harness.scope["_simulator_cpu_topology"].cache_info().currsize, 1
        )

    def test_public_parser_always_returns_a_fresh_uncached_list(self):
        harness = TopologyHarness()
        parser = harness.scope["parse_lscpu_topology"]
        first = parser()
        first.clear()
        self.assertEqual(len(parser()), 5)
        self.assertEqual(harness.read_calls, 2)

    def test_parser_keeps_empty_defaults_and_skips_incomplete_rows(self):
        harness = TopologyHarness()
        harness.csv = "# topology\n0,,,\nincomplete\n1,1,0,0\n"
        with self.assertLogs(__name__, level="WARNING") as logs:
            self.assertEqual(harness.physical(), {0: {0, 1}})
        self.assertIn("Skipping malformed lscpu line", logs.output[0])

    def test_unreadable_online_file_keeps_original_query_behavior(self):
        harness = TopologyHarness()
        harness.online_error = OSError("not mounted")
        self.assertEqual(harness.physical(), harness.physical())
        self.assertEqual(harness.read_calls, 2)

    def test_subprocess_errors_are_not_cached(self):
        harness = TopologyHarness()
        harness.read_error = OSError("unavailable")
        for _ in range(2):
            with self.assertRaisesRegex(
                RuntimeError, "Unexpected error running 'lscpu'"
            ):
                harness.physical()
        self.assertEqual(harness.read_calls, 2)
        self.assertEqual(
            harness.scope["_simulator_cpu_topology"].cache_info().currsize, 0
        )
        harness.read_error = None
        self.assertEqual(harness.physical(), {0: {0, 1}, 1: {4, 5}})

    def test_malformed_numeric_topology_is_not_retained(self):
        harness = TopologyHarness()
        harness.csv = "bad,0,0,0\n"
        for _ in range(2):
            with self.assertRaises(ValueError):
                harness.physical()
        self.assertEqual(harness.read_calls, 2)
        harness.csv = harness.CSV
        self.assertEqual(harness.physical(), {0: {0, 1}, 1: {4, 5}})

    def test_replaced_parser_bypasses_cached_topology(self):
        harness = TopologyHarness()
        harness.physical()
        calls = []

        def parser():
            calls.append(None)
            return [(5, 1, 1, 1)]

        harness.scope["parse_lscpu_topology"] = parser
        self.assertEqual(harness.physical(), {1: {5}})
        self.assertEqual(harness.physical(), {1: {5}})
        self.assertEqual(len(calls), 2)

    def test_replaced_subprocess_reader_bypasses_cached_topology(self):
        harness = TopologyHarness()
        harness.physical()
        calls = []

        def read_csv(*args, **kwargs):
            calls.append(None)
            return "5,1,1,1\n"

        harness.scope["subprocess"].check_output = read_csv
        self.assertEqual(harness.physical(), {1: {5}})
        self.assertEqual(harness.physical(), {1: {5}})
        self.assertEqual(len(calls), 2)

    def test_disabling_opt_in_restores_fresh_source_reads(self):
        harness = TopologyHarness()
        harness.physical()
        harness.env.clear()
        harness.csv = "5,1,1,1\n"
        self.assertEqual(harness.physical(), {1: {5}})
        self.assertEqual(harness.physical(), {1: {5}})
        self.assertEqual(harness.read_calls, 3)


if __name__ == "__main__":
    unittest.main()
