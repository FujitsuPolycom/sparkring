"""Build the DCP2 arm's `docker run` argv for one rank from the DCP1 control's installer container.

Usage: dcp2_argv.py INSPECT.json RANK RECEIPTS_HOSTDIR > argv.json

The installer renders glm53-flash-nvfp4-spark-tp2-dcp2 as the pair profile's csf deployment with exactly these
changes (runtime/common/test_transport.py,
test_the_pair_dcp2_variant_is_the_pair_profiles_csf_deployment_with_its_kv_cache_split):
--served-model-name GLM-5.3-Flash-CSF-DCP2-TP2, --kv-cache-memory-bytes 15032385536,
--decode-context-parallel-size 2 followed by --cp-kv-cache-interleave-size 4, SIRCL_GROUPS=tp,dcp,
VLLM_GLM53_MHC_PREFILL_SHARD=0,
VLLM_B12X_KDA_PREFILL_COALESCING=0 and the cache namespace glm53-flash-csf-dcp2 in every /cache path. This program
applies those changes to the running DCP1 container's settings and otherwise reproduces them; it also gives the
container its own name and labels (no io.sparkring.* label, so SparkRing's installer does not treat it as its
deployment), its own SIRCL receipt directory, and no runtime binding (an identity file of the installer's
deployment, read only by the status plugin). Anything it does not reproduce is printed on stderr.
"""
import json
import sys

record = json.load(open(sys.argv[1]))
record = record[0] if isinstance(record, list) else record
rank, receipts = int(sys.argv[2]), sys.argv[3]
cfg, host = record["Config"], record["HostConfig"]
NAMESPACE_OLD, NAMESPACE_NEW = "/cache/glm53-flash-nvfp4-spark-", "/cache/glm53-flash-csf-dcp2-"
argv = ["docker", "run", "-d", "--name", f"glm53-dcp2-r{rank}", "--label", "glm53-dcp2-research=20261013",
        "--label", f"glm53-dcp2-research.rank={rank}"]
if host.get("NetworkMode") not in (None, "default", "bridge"):
    argv += ["--network", host["NetworkMode"]]
if host.get("IpcMode") in ("host",):
    argv += ["--ipc", "host"]
if host.get("PidMode"):
    argv += ["--pid", host["PidMode"]]
if host.get("ShmSize") not in (None, 0, 67108864):
    argv += ["--shm-size", str(host["ShmSize"])]
if host.get("Privileged"):
    argv.append("--privileged")
for key, opt in (("CapAdd", "--cap-add"), ("CapDrop", "--cap-drop"), ("SecurityOpt", "--security-opt"),
                 ("GroupAdd", "--group-add")):
    for value in host.get(key) or []:
        if key == "SecurityOpt" and value.startswith("seccomp={"):
            # docker inspect shows the seccomp profile's content; docker run reads it from a file on the Spark.
            import os
            with open(os.environ["SECCOMP_OUT"], "w", newline="") as handle:
                handle.write(value[len("seccomp="):])
            value = "seccomp=" + os.environ["SECCOMP_PATH"]
        argv += [opt, value]
for limit in host.get("Ulimits") or []:
    argv += ["--ulimit", f"{limit['Name']}={limit['Soft']}:{limit['Hard']}"]
for device in host.get("Devices") or []:
    argv += ["--device", f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"]
for request in host.get("DeviceRequests") or []:
    if request.get("Count") == -1 and ["gpu"] in (request.get("Capabilities") or []):
        argv += ["--gpus", "all"]
    else:
        print("NOT REPRODUCED DeviceRequest", request, file=sys.stderr)
if host.get("Binds"):
    print("NOT REPRODUCED Binds", host["Binds"], file=sys.stderr)
receipt_target = None
for env in cfg.get("Env") or []:
    if env.startswith("SIRCL_RECEIPT_DIR="):
        receipt_target = env.split("=", 1)[1]
for mount in record.get("Mounts") or []:
    source, target = mount["Source"], mount["Destination"]
    if target == "/run/sparkring/runtime-binding.json":
        continue
    if target == receipt_target:
        source = receipts
    option = f"type={mount['Type']},source={source},target={target}"
    if not mount.get("RW", True):
        option += ",readonly"
    argv += ["--mount", option]
for key, opt in (("Memory", "--memory"), ("CpusetCpus", "--cpuset-cpus")):
    if host.get(key):
        argv += [opt, str(host[key])]
if host.get("Init"):
    argv.append("--init")
environment = {}
for env in cfg.get("Env") or []:
    key, _, value = env.partition("=")
    environment[key] = value
environment.pop("SPARKRING_RUNTIME_BINDING", None)
environment["SIRCL_GROUPS"] = "tp,dcp"
environment["VLLM_GLM53_MHC_PREFILL_SHARD"] = "0"
# The image's vLLM admits KDA prefill coalescing at TP2 only with DCP1 (validate_coalescing_config in
# vllm/v1/core/recurrent_prefill_checkpoint.py); the profile turns it off.
environment["VLLM_B12X_KDA_PREFILL_COALESCING"] = "0"
for key, value in environment.items():
    if value.startswith(NAMESPACE_OLD):
        environment[key] = NAMESPACE_NEW + value[len(NAMESPACE_OLD):]
for key, value in environment.items():
    argv += ["-e", f"{key}={value}"]
if cfg.get("WorkingDir"):
    argv += ["-w", cfg["WorkingDir"]]
if cfg.get("User"):
    argv += ["-u", cfg["User"]]
command = list(cfg.get("Cmd") or [])


def set_value(flag, value):
    command[command.index(flag) + 1] = value


set_value("--served-model-name", "GLM-5.3-Flash-CSF-DCP2-TP2")
set_value("--kv-cache-memory-bytes", sys.argv[4] if len(sys.argv) > 4 else "15032385536")
if len(sys.argv) > 5:
    set_value("--max-model-len", sys.argv[5])
assert command[command.index("--decode-context-parallel-size") + 1] == "1"
assert "--cp-kv-cache-interleave-size" not in command
position = command.index("--decode-context-parallel-size")
command[position + 1:position + 2] = ["2", "--cp-kv-cache-interleave-size", "4"]
entry = cfg.get("Entrypoint") or []
if entry:
    argv += ["--entrypoint", entry[0]]
argv.append(record["Image"])
argv += entry[1:] + command
restart = (host.get("RestartPolicy") or {}).get("Name")
if restart not in (None, "", "no"):
    print(f"NOT REPRODUCED RestartPolicy {restart} (the research container does not restart)", file=sys.stderr)
for key in ("PortBindings", "ExtraHosts", "VolumesFrom", "Links", "Tmpfs", "Sysctls", "NanoCpus"):
    if host.get(key) not in (None, [], {}, 0, ""):
        print(f"NOT REPRODUCED HostConfig.{key}={json.dumps(host[key])}", file=sys.stderr)
if cfg.get("Healthcheck"):
    print("NOT REPRODUCED Healthcheck (the API is polled directly)", file=sys.stderr)
print(json.dumps(argv, indent=1))
