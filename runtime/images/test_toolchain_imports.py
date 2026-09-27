"""GPU-free installed-image regression for explicit vendor-library loading."""
from pathlib import Path
import subprocess
import sys

import pytest


TOOLCHAIN = Path('/opt/sparkring/toolchain/toolchain.py')


@pytest.mark.skipif(not TOOLCHAIN.is_file(), reason='Requires installed candidate toolchain')
@pytest.mark.parametrize('modules', [('av', 'soundfile', 'torch', 'vllm', 'b12x'),
                                    ('torch', 'b12x', 'vllm', 'av', 'soundfile')])
def test_imports_load_only_selected_libraries(modules):
    # A subprocess is essential: dlopen mappings outlive imports, and the ELF
    # loader reads preloads before Python starts. No GPU initialization is used.
    program = r'''
import importlib, importlib.util, json, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('toolchain', '/opt/sparkring/toolchain/toolchain.py')
toolchain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toolchain)
lock = json.loads(Path('/opt/sparkring/toolchain/toolchain.json').read_text())
environment = toolchain.configure_environment(lock, os.environ)
if any(os.environ.get(key) != value for key, value in environment.items()):
    os.execve(sys.executable, [sys.executable, '-c', sys.argv[1], *sys.argv[1:]], environment)
for name in sys.argv[2:]:
    importlib.import_module(name)
    toolchain.verify_loaded_libraries(lock, Path('/proc/self/maps').read_text())
'''
    result = subprocess.run([sys.executable, '-c', program, program, *modules],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
