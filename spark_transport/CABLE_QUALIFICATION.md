# Direct-cable qualification

Status: **implemented** offline qualification procedure.

Run `spark_transport/scripts/qualify_direct_cable.py` on every new, moved, or
suspect direct link before starting a distributed workload. Link-up and ping
are insufficient: qualification requires bidirectional payload integrity and
stable NIC error counters.

The tool supports 12,288-byte and 16,384-byte workload descriptors. Run the
descriptor selected by the deployment profile, or both when qualifying a cable
for general cluster use. The [profile registry](../docs/profiles/README.md)
records model-specific descriptor meaning.

## Safety boundary

The controller is read-only. It:

- reads link, address, route, RDMA/GID, and NIC-counter state over SSH;
- sends five interface-bound pings in each direction;
- verifies the hash of an already installed probe on both endpoints;
- runs the selected payload probes in both directions; and
- reads the same counters afterward.

It never changes an address, route, MTU, queue discipline, offload, interrupt
binding, driver binding, or serving process. A preflight mismatch stops the
payload test.

## Result contract

Progress is written to stderr. One versioned JSON document is written to
stdout and, when requested, to `--output`.

| Exit | Meaning |
|---:|---|
| 0 | Bidirectional integrity passed. A software-latency warning may remain. |
| 1 | A link, integrity, counter, configuration, or strict-latency check failed. |
| 2 | Input, SSH, tool, timeout, or probe-output error. |
| 3 | Preflight passed without a payload test; the cable is not qualified. |

Important fields are:

- `cable_qualified`: payload integrity passed with no new physical-link errors;
- `latency_target_met`: the userspace path met its configured p99 target;
- `model_path_ready`: integrity passed and any requested strict latency target passed;
- `failure_domain`: classifies physical, configuration, software, and orchestration failures; and
- `counter_deltas`: separates physical-link errors from software pressure.

Cable integrity and software latency are separate results. Zero loss and zero
counter changes can qualify the cable even when the tested software path misses
its latency target. Any CRC error, missing fragment, lost message, kernel drop,
or new physical-link error rejects integrity regardless of median latency.

## 200 Gb/s RoCE example

Build `spark_transport_probe` with the normal Spark Transport CMake build and
install the same executable at `/tmp/spark_transport_probe` on both systems.
The controller verifies SHA-256 equality.

```bash
python3 spark_transport/scripts/qualify_direct_cable.py \
  --tier roce200 \
  --left user@192.0.2.1 \
  --right user@192.0.2.2 \
  --left-interface enp1s0f0np0 \
  --right-interface enp1s0f0np0 \
  --left-ip 198.18.0.10 \
  --right-ip 198.18.0.11 \
  --left-rdma-device rocep1s0f0 \
  --right-rdma-device rocep1s0f0 \
  --gid-index 3 \
  --expected-mtu 9000 \
  --probe-binary /tmp/spark_transport_probe \
  --payloads 12288,16384 \
  --iterations 10000 \
  --strict-latency \
  --output results/cable-rank0-rank1.json
```

The RoCE tier requires the expected 200,000 Mb/s link rate, interface address,
direct route, active RDMA port, RoCEv2 GID, MTU, and bidirectional reliable-
connection writes. Its default p99 target is 20 microseconds. Override the
target only as an explicitly recorded experiment.

## Retest policy

`--preflight-only` helps identify ports but always exits 3. Save one JSON result
for every direct edge and rerun an affected edge after changing a cable, NIC,
address, route, MTU, GID, firmware, or driver, or after an unexplained
collective hang or tail-latency increase.

Do not average directions. An asymmetric integrity failure rejects the link.
Do not compare latency until every integrity check passes.
