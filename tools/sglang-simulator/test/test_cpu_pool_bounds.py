"""Generic CPU pool inversion bounds; no SDK or resource-provider selection."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def profile_budget(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "src/sglang_simulator/simulation/sglang/model_runner.py"
    )
    tree = ast.parse(source.read_text())
    resolve = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_kv_page_size"
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "C_KVCacheConfiguratorHook"
    )
    hook = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "hook"
    )
    profile = next(
        node
        for node in hook.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "override_profile_available_bytes"
    )
    # P4 may extract the same ordinary search into a helper when adding its
    # optional resource-budget branch. Exercise the actual ordinary path in
    # either shape without importing optional runtime dependencies.
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_native_pool_budget"
    ]
    namespace = {
        "ConfigManager": SimpleNamespace(set_memory_budget_metrics=lambda _: None)
    }
    exec(
        compile(
            ast.Module(body=[resolve, *helpers, profile], type_ignores=[]),
            str(source),
            "exec",
        ),
        namespace,
    )
    module_name = "sglang.srt.model_executor.pool_configurator"
    native = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, native)

    def run(pool, tokens, page_size=4):
        native.create_memory_pool_configurator = lambda _: pool
        configurator = SimpleNamespace(
            server_args=SimpleNamespace(max_total_tokens=tokens), page_size=page_size
        )
        return namespace["override_profile_available_bytes"](configurator, 0)

    return run


class LinearPool:
    def __init__(self):
        self.calls = 0

    def calculate_pool_sizes(self, budget_bytes, page_size):
        self.calls += 1
        if budget_bytes < 11:
            raise ValueError("cannot hold even one request")
        tokens = (budget_bytes - 11) // 3
        return SimpleNamespace(max_total_num_tokens=tokens // page_size * page_size)


def test_explicit_capacity_uses_native_pool_calculation(profile_budget):
    assert profile_budget(LinearPool(), 16) == 59


def test_non_aligned_target_keeps_existing_at_least_capacity_rule(profile_budget):
    assert profile_budget(LinearPool(), 13) == 59


def test_unreachable_layout_stops_after_finite_probes(profile_budget):
    class ConstantPool:
        calls = 0

        def calculate_pool_sizes(self, budget_bytes, page_size):
            self.calls += 1
            return SimpleNamespace(max_total_num_tokens=8)

    pool = ConstantPool()
    with pytest.raises(ValueError, match="Native KV pool cannot represent"):
        profile_budget(pool, 16)
    assert pool.calls == 64


def test_undersized_runtime_errors_keep_existing_probe_behavior(profile_budget):
    class Pool(LinearPool):
        def calculate_pool_sizes(self, budget_bytes, page_size):
            if budget_bytes < 11:
                raise RuntimeError("undersized probe")
            return super().calculate_pool_sizes(budget_bytes, page_size)

    assert profile_budget(Pool(), 13) == 59


def test_other_value_errors_are_not_treated_as_small_budgets(profile_budget):
    class InvalidPool:
        def calculate_pool_sizes(self, budget_bytes, page_size):
            raise ValueError("invalid layout")

    with pytest.raises(ValueError, match="invalid layout"):
        profile_budget(InvalidPool(), 16)
