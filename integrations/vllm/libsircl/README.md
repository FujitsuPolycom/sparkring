# vLLM plugin `libsircl`

Status: **research-only**. [sparkring_libsircl.py](sparkring_libsircl.py) is
a vLLM general plugin that makes vLLM's PyNccl load libsircl
([spark_transport/libsircl](../../../spark_transport/libsircl/README.md)), SIRCL's
NCCL-compatible C library, in an installer image.

vLLM's PyNccl loads the library that `VLLM_NCCL_SO_PATH` names. The installer
image's entrypoint sets that variable to the image's NVIDIA NCCL before vLLM
starts, so the plugin sets it inside vLLM's processes instead:

| Variable | Meaning |
|---|---|
| `VLLM_PLUGINS` | names `libsircl` to load the plugin |
| `SPARKRING_LIBSIRCL_LIBRARY` | absolute path of the library, a regular file |
| `SPARKRING_LIBSIRCL_SHA256` | the library's SHA-256 |

vLLM calls `register` in each process before that process creates a PyNccl
communicator. `register` checks the file against its SHA-256 and sets
`VLLM_NCCL_SO_PATH` to it; a missing variable, a missing or different file
fails the process. torch's ProcessGroupNCCL keeps the NCCL its process loaded
at start.

The [libsircl image layer](../../../runtime/images/libsircl_layer.py) installs
the module in the serving interpreter's site-packages with a dist-info
directory whose `entry_points.txt` registers it in `vllm.general_plugins`,
and the [libsircl transport](../../../runtime/common/libsircl.py) sets the
three variables. [docs/architecture/libsircl.md](../../../docs/architecture/libsircl.md)
describes both.

```bash
python -m pytest integrations/vllm/libsircl -q
```
