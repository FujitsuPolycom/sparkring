# Direct-cable qualification

Status: implemented qualification controller with offline contract tests.
Execution contacts both hosts and generates network and RDMA traffic.

Run `spark_transport/scripts/qualify_direct_cable.py` on every new, moved, or suspect
direct-attached cable **before** loading a model. Link-up and ping are not
enough: by default, the test checks bidirectional traffic for the 12,288-byte width-6144 Q1
BF16 payload (`[1, 6144]`) and the 16,384-byte width-4096 Q2 BF16 payload
(`[2, 4096]`) while watching NIC error counters.
`--payloads` can select either size; the result qualifies only the selected
sizes, which are recorded in the JSON report.

The controller is deliberately non-destructive. It only:

- reads link, address, route, RDMA/GID, and NIC-counter state over SSH;
- sends five source-address-bound pings in each direction;
- hashes an already-installed probe on both endpoints;
- runs the payload probe in both directions; and
- reads the same counters afterward.

It never changes an address, route, MTU, qdisc, offload, IRQ, driver binding,
or model process. A preflight mismatch stops the payload test.

## Exit codes and JSON

Progress goes to stderr. One versioned JSON document goes to stdout and,
optionally, to `--output`.

| Exit | Meaning |
|---:|---|
| 0 | Cable integrity qualified for the selected payload sizes. The JSON may still contain a software-latency warning. |
| 1 | A hard link, integrity, counter, configuration, or strict-latency gate failed. |
| 2 | Invalid input, SSH/tool error, timeout, or malformed probe output. |
| 3 | Preflight passed but no payload test ran; the cable is **not** qualified. |

Important result fields are:

- `cable_qualified`: bidirectional payload integrity passed with no new
  PHY/cable error counters;
- `latency_target_met`: the selected userspace transport met its p99 target;
- `model_path_ready`: cable is qualified and, with `--strict-latency`, the
  latency target also passed;
- `failure_domain`: distinguishes `cable_or_phy`, `configuration`,
  `software_latency`, `software_pressure`, and orchestration failures;
- `counter_deltas`: separates CRC/alignment/carrier/MAC errors from
  drop/miss/overrun pressure.

Qualification requires readable error counters before traffic. RoCE also
requires a successful `ethtool -S` query with hardware error counters; generic
netdev counters alone do not cover RDMA traffic. Disappearing counters or
newly visible PHY counters without a baseline fail the instrumentation gate.
The retained `reset` category includes these telemetry discontinuities.

The distinction matters. A raw 10GbE run with zero loss/CRC/counter deltas
but 160 microseconds p99 has proven the cable; it has **not** proven that the
AF_PACKET/NAPI software path is suitable for decode. Conversely, any CRC,
missing fragment, lost message, kernel drop, or new PHY error is an integrity
failure regardless of a good median.

## 200G RoCE cable

Build `spark_transport_probe` with the normal Spark Transport CMake build and
install the exact same executable at `/tmp/spark_transport_probe` on both
Sparks. The script verifies SHA-256 equality.

Example for the rank 0--1 edge (`<SUBNET_01>` is that cable's /24 fabric
prefix from the private site inventory):
```bash
python3 spark_transport/scripts/qualify_direct_cable.py \
  --tier roce200 \
  --left user@192.0.2.1 \
  --right user@192.0.2.2 \
  --left-interface enp1s0f0np0 \
  --right-interface enp1s0f1np1 \
  --left-ip <SUBNET_01>.10 \
  --right-ip <SUBNET_01>.11 \
  --left-rdma-device rocep1s0f0 \
  --right-rdma-device rocep1s0f1 \
  --gid-index 3 \
  --expected-mtu 9000 \
  --probe-binary /tmp/spark_transport_probe \
  --iterations 10000 \
  --strict-latency \
  --output results/cable-spark0-spark1.json
```

The RoCE tier additionally requires:

- exactly 200,000 Mb/s on both ports;
- the expected IP and direct route on the named interfaces;
- active RDMA ports;
- the selected GID index bound to the named netdev as `RoCE v2`; and
- verified host-memory RC writes for each selected payload size in both
  directions, with matching byte/sample counts and one result record per endpoint.

The default p99 target is 20 microseconds. Override it only as an explicit
experiment with `--max-p99-us`.

## Fast preflight and test policy

`--preflight-only` is useful while identifying an unlabeled port. A passing
preflight returns exit 3; failed checks retain exit 1 or 2. Link state and
ping cannot qualify a cable.

For a four-Spark RoCE cycle, save one JSON result for each of its four
200G edges. Additional physical diagonal links are not required by that
topology. Qualify any additional links required by a separately selected
deployment. Re-run the affected edge after:

- changing or reseating a cable;
- changing NIC, IP, route, MTU, GID, firmware, or driver;
- unexplained collective hangs, retransmission, or tail-latency growth; or
- moving the cluster.

Do not average directions. An asymmetric failure is a cable/link failure.
Do not compare latency until all integrity gates pass.

Recovery does **not** qualify a cable. After a reseat, asymmetric
negotiation, or recovery action, rerun this full bidirectional qualification
before loading a model.
