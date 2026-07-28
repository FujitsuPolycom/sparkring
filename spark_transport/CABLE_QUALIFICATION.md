# Direct-cable qualification

Run `scripts/qualify_direct_cable.py` on every new, moved, or suspect
direct-attached cable **before** loading a model. Link-up and ping are not
enough: the test requires bidirectional 12,288-byte and 16,384-byte traffic,
the two latency-sensitive GLM collective sizes, while watching NIC error
counters.

The controller is deliberately non-destructive. It only:

- reads link, address, route, RDMA/GID, and NIC-counter state over SSH;
- sends five interface-bound pings in each direction;
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
| 0 | Cable integrity qualified. The JSON may still contain a software-latency warning. |
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
prefix from `../docs/SETUP.md` Section 1):

```bash
python3 spark_transport/scripts/qualify_direct_cable.py \
  --tier roce200 \
  --left user@192.0.2.1 \
  --right user@192.0.2.2 \
  --left-interface enp1s0f0np0 \
  --right-interface enp1s0f0np0 \
  --left-ip <SUBNET_01>.10 \
  --right-ip <SUBNET_01>.11 \
  --left-rdma-device rocep1s0f0 \
  --right-rdma-device rocep1s0f0 \
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
- GID index 3 bound to the named netdev as `RoCE v2`; and
- verified RC writes at 12 KB and 16 KB in both directions.

The default p99 target is 20 microseconds. Override it only as an explicit
experiment with `--max-p99-us`.

## Direct 10GbE diagonal

Build the existing raw benchmark:

```bash
cd spark_transport/experiments/ten_gbe_diagonal
./build_raw.sh
./ten_gbe_raw_bench --self-test
```

Copy the same executable to `/tmp/ten_gbe_raw_bench` on both endpoints. Raw
Ethernet requires `CAP_NET_RAW`; `--use-sudo` invokes only the benchmark with
`sudo -n` and will fail closed if noninteractive sudo is unavailable.

Example for rank 0--2 with its current MTU 1500 (`<SUBNET_D02>` is the
diagonal's own /24 prefix, distinct from the four ring `<SUBNET_xx>` prefixes
in `../docs/SETUP.md`):

```bash
python3 spark_transport/scripts/qualify_direct_cable.py \
  --tier diagonal10 \
  --left user@192.0.2.1 \
  --right user@192.0.2.3 \
  --left-interface enP7s7 \
  --right-interface enP7s7 \
  --left-ip <SUBNET_D02>.1 \
  --right-ip <SUBNET_D02>.2 \
  --expected-mtu 1500 \
  --probe-binary /tmp/ten_gbe_raw_bench \
  --left-cpu 13 \
  --right-cpu 13 \
  --use-sudo \
  --iterations 10000 \
  --output results/cable-r0-r2-10gbe.json
```

Use `--expected-mtu 9000` for a diagonal already configured and verified at
jumbo MTU. The script does not change it.

The raw benchmark uses private EtherType `0x88B5`, TPACKET_V2, `sendmmsg`,
exact MAC filtering, per-fragment and full-payload CRC, and sequence/loss
accounting. The default 30-microsecond p99 target is a *transport* target. A
miss is a warning by default because AF_PACKET includes kernel/NAPI work; add
`--strict-latency` only when this cable must qualify for the current decode
critical path.

## Fast preflight and test policy

`--preflight-only` is useful while identifying an unlabeled port, but always
returns exit 3: link state and ping cannot qualify a cable.

For a four-Spark installation, save one JSON result for each of the four
200G cycle edges and both 10GbE diagonals. Re-run the affected edge after:

- changing or reseating a cable;
- changing NIC, IP, route, MTU, GID, firmware, or driver;
- unexplained collective hangs, retransmission, or tail-latency growth; or
- moving the cluster.

Do not average directions. An asymmetric failure is a cable/link failure.
Do not compare latency until all integrity gates pass.

For an unreachable rank 0--2 `enP7s7` link, the fail-closed remote service
recovery procedure is documented in
`experiments/ten_gbe_diagonal/RECOVERY.md`.
Recovery does **not** qualify the cable. After a reseat, asymmetric
negotiation, or recovery action, rerun this full bidirectional qualification
before loading a model.
