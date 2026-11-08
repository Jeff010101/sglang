"""Fresh-process CPU boundary controls without accelerator dependencies.

These controls exercise import and delegation contracts. Real SGLang/torch
quantization validation is covered separately by test_cpu_bootstrap.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SIMULATOR = Path(__file__).resolve().parents[1]
REPOSITORY = SIMULATOR.parents[1]


def run_isolated(code, source, *arguments):
    env = os.environ.copy()
    env.pop("SGLANG_SIMULATOR_BOOTSTRAP", None)
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(source), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "bootstrap,cpu", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fresh_process_cpu_flags_and_repeat_activation(bootstrap, cpu):
    run_isolated(
        """
import os, runpy, sys
from types import ModuleType, SimpleNamespace

def available():
    return True

def capability(*args):
    return (9, 0)

torch = ModuleType('torch')
torch.cuda = SimpleNamespace(is_available=available, get_device_capability=capability)
torch.version = SimpleNamespace(hip='original-rocm')
sys.modules['torch'] = torch
modules = {}
for name in ('sglang', 'sglang.srt', 'sglang.srt.utils', 'sglang.srt.utils.common',
             'sglang_simulator', 'sglang_simulator.simulation',
             'sglang_simulator.simulation.sglang',
             'sglang_simulator.simulation.sglang.sgl_kernel_hook'):
    module = ModuleType(name)
    module.__path__ = []
    modules[name] = sys.modules[name] = module
    if '.' in name:
        parent, leaf = name.rsplit('.', 1)
        setattr(modules[parent], leaf, module)
utils = modules['sglang.srt.utils']
common = modules['sglang.srt.utils.common']
utils.get_device_capability = common.get_device_capability = capability
calls = []
modules['sglang_simulator.simulation.sglang.sgl_kernel_hook'].install_load_utils_stub = lambda: calls.append('loader')
if sys.argv[2] == '1':
    os.environ['SGLANG_SIMULATOR_BOOTSTRAP'] = '1'
if sys.argv[3] == '1':
    os.environ['SGLANG_USE_CPU_ENGINE'] = '1'
namespace = runpy.run_path(sys.argv[1])
namespace['apply_cpu_simulation_compat']()
enabled = sys.argv[2:] == ['1', '1']
if enabled:
    assert torch.cuda.is_available() is False
    assert torch.cuda.get_device_capability() == (10, 0)
    assert torch.version.hip is None
    assert utils.get_device_capability() == common.get_device_capability() == (10, 0)
    assert calls == ['loader', 'loader']
else:
    assert torch.cuda.is_available is available
    assert torch.cuda.get_device_capability is capability
    assert torch.version.hip == 'original-rocm'
    assert utils.get_device_capability is common.get_device_capability is capability
    assert calls == []
assert 'sglang.srt.layers.quantization' not in sys.modules
""",
        SIMULATOR / "src/usercustomize.py",
        str(int(bootstrap)),
        str(int(cpu)),
    )


def test_quark_defers_aiter_and_delegates_actual_call():
    run_isolated(
        """
import importlib.abc, runpy, sys
from types import ModuleType, SimpleNamespace

torch = ModuleType('torch')
torch.Tensor = object
torch.nn = SimpleNamespace(Module=object)
sys.modules['torch'] = torch

class BlockAiter(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'aiter' or fullname.startswith('aiter.'):
            raise ImportError('AITER intentionally absent')

blocker = BlockAiter()
sys.meta_path.insert(0, blocker)
namespace = runpy.run_path(sys.argv[1])
assert 'aiter' not in sys.modules
events = []

class Input:
    shape = (2, 3, 64)
    def reshape(self, *shape):
        events.append(('reshape', shape))
        return self

try:
    namespace['b_dynamic_mxfp4_quant'](Input())
except ImportError as error:
    assert 'Failed to import aiter' in str(error)
    assert isinstance(error.__cause__, ImportError)
else:
    raise AssertionError('the actual call must require AITER')
assert events == []
sys.meta_path.remove(blocker)

class Result:
    def __init__(self, name):
        self.name = name
    def view(self, *shape):
        events.append((self.name, shape))
        return (self.name, shape)

def quantize(value):
    assert isinstance(value, Input)
    events.append(('quantize',))
    return Result('packed'), Result('scales')

for name in ('aiter', 'aiter.ops', 'aiter.ops.triton', 'aiter.ops.triton.quant'):
    module = ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
sys.modules['aiter.ops.triton.quant'].dynamic_mxfp4_quant = quantize
assert namespace['b_dynamic_mxfp4_quant'](Input()) == (
    ('packed', (2, 3, 32)), ('scales', (2, 3, 2)))
assert events == [('reshape', (-1, 64)), ('quantize',),
                  ('packed', (2, 3, 32)), ('scales', (2, 3, 2))]
""",
        REPOSITORY / "python/sglang/srt/layers/quantization/quark/utils.py",
    )
