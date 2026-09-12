"""Check the C diagnostic counter paths under ThreadSanitizer without RDMA/GPU access."""
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux GCC ThreadSanitizer check")
def test_proxy_diagnostic_counter_reads_do_not_race(tmp_path):
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("GCC is unavailable")
    root = Path(__file__).resolve().parents[3]
    source = (root / "third_party/b12x_roce/b12x/comm/roce/_roce_proxy.c").read_text()
    fields = re.findall(
        r"^\s+(?:atomic_uint_\w+_t|uint\d+_t) (?:last_seq|ops_posted|writes_completed|two_wave_activations);$",
        source, re.MULTILINE,
    )
    assert len(fields) == 4
    start = source.index("uint64_t roce_stat(")
    end = source.index("\nuint64_t roce_two_wave_threshold_bytes", start)
    # Retain the actual declarations and accessor. Only the context's unused
    # RDMA fields are omitted; the test never constructs a transport context.
    program = (
        "#include <stdint.h>\n#include <stdatomic.h>\n#include <pthread.h>\n"
        + "typedef struct {\n" + "\n".join(fields) + "\n} roce_ctx_t;\n"
        + source[start:end]
        + r'''
static roce_ctx_t context;
static pthread_barrier_t barrier;
static void *writer(void *argument) {
 int which = *(int *)argument;
 pthread_barrier_wait(&barrier);
 for (uint32_t i = 1; i <= 100000; ++i) {
  if (which == 0) context.ops_posted += 1;
  else if (which == 1) context.writes_completed += 1;
  else context.last_seq = i;
 }
 return 0;
}
int main(void) {
 if (pthread_barrier_init(&barrier, 0, 2)) return 2;
 for (int which = 0; which < 3; ++which) {
  pthread_t thread;
  if (pthread_create(&thread, 0, writer, &which)) return 3;
  pthread_barrier_wait(&barrier);
  volatile uint64_t observed = 0;
  for (int i = 0; i < 100000; ++i) observed = roce_stat(&context, which);
  (void)observed;
  pthread_join(thread, 0);
  if (roce_stat(&context, which) != 100000) return 4;
 }
 pthread_barrier_destroy(&barrier);
 return 0;
}
'''
    )
    path = tmp_path / "counter_paths.c"
    path.write_text(program)
    binary = tmp_path / "counter_paths"
    build = subprocess.run(
        [compiler, "-std=gnu11", "-O1", "-g", "-fsanitize=thread", "-no-pie",
         "-pthread", str(path), "-o", str(binary)],
        capture_output=True, text=True, timeout=30,
    )
    assert build.returncode == 0, build.stderr
    command = [str(binary)]
    if shutil.which("setarch"):
        command = ["setarch", platform.machine(), "-R", *command]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if "unexpected memory mapping" in result.stderr or "failed to set personality" in result.stderr:
        pytest.skip("Host cannot provide the address layout required by ThreadSanitizer")
    assert result.returncode == 0, result.stderr
    assert "ThreadSanitizer:" not in result.stderr
