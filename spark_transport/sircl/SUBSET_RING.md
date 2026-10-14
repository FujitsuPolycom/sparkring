# Subset chain all-gather

**Experimental.** Set `SIRCL_SUBSET_RING=1` and `SIRCL_ROUTE_SCOPE=fabric`
on the tensor-parallel session, or use the serve/bundle option `--subset-ring`.
The public option retains its name for compatibility. It is off by default.

On an admitted subset, only flat (1-D) `all_gather_large` can use chain.
The input shard must be at least 4 MiB, satisfy the existing alignment/shape
checks, and meet the configured chain minimum for automatic selection.
Explicit `pieces` disables it. A `ring` gather request becomes `auto`.
All-reduce, reduce-scatter, row, padded and 2-D gathers use the pieces path;
small collectives retain their existing one-shot/two-shot behavior. Runtime
schedule overrides and tuning choices cannot enable subset ring operations.
Every native ring neighbor is disabled and its window is zero.

Admission requires a cable-connected arc of at least three ranks on a cycle
of four to eight Sparks, leaving one or two adjacent non-members. Wrapping
arcs and reordered ranks are supported. Larger complements are refused,
including when idle. Use the universal same-function relay table of the
complete cycle; setup verifies every lane against that table. The mode
does not install or modify fabric routes. Full-fabric sessions retain their
existing behavior even when the option is set.

## Selection evidence

The six-rank measurement of `cb519293` on ring8 positions 0-5, with DCP2
on 6-7, found flat all-gather chain throughput at 1.20x OFF for 4 MiB
shards and 1.32x for 16 MiB shards. Those are hardware measurements of that
revision, not qualification of this policy. The 4 MiB floor avoids enabling
chain for the small and 1 MiB cases. Other sizes at or above that floor and
other admitted placements require hardware confirmation.

Reduce-scatter can traverse the same direct chain, but its two partial
reductions, scratch storage and rendezvous differ from forwarding gather
shards. All-reduce also performs reduction and return traffic. Direct cables
alone do not establish a speedup for either collective; both remain on pieces.

The measured padded/2-D gather regression occurred despite unchanged 2 MiB
pieces. The former opt-in changed the underlying transport: chunks shrank
from 32 KiB to 4 KiB, the common per-queue allowance shrank from 393,216 to
90,112 bytes, and delivery-proof credit release was disabled. On the busiest
five-lane queues, per-lane windows shrank from 65,536 to 16,384 bytes.
More chunk posts and longer credit waits can slow the same GPU pieces.
The measurement changed these settings together and does not isolate their
individual costs. This mode now uses exactly the OFF pieces sizes, forward
windows, transport chunks and native ACK-release policy, including explicit
user overrides. Recovery of throughput still needs hardware verification.

## Capacity and isolation

The active chain consists solely of direct neighbor lanes in both directions.
It adds **zero data or ACK bytes** to every relay hairpin queue. There is no
closing edge, ring-window reservation or relayed link operation. Sessions
confined to the one/two adjacent non-members likewise add zero relay bytes.
These facts hold independently of traffic rate and are checked across the
admitted layouts. Direct traffic still competes for cables and NIC resources;
DCP2 decode isolation must be measured with no unrelated clients.

Pieces keeps the OFF capacity model: default forward windows share 75% of
the declared queue, rounded to 32 KiB chunks; disabling windows uses the
existing relay-load piece limit. On the admitted default lane maps the sum
of windows is at most 393,216 bytes per 524,288-byte queue. Chain adds zero
to that bound. The mode continues reserving the full collective share from
the same-instance P2P allocator, preventing extra relayed P2P sessions.
Overlapping independent deployments on the member positions are not covered.

This is a payload/delivery bound inherited from OFF, **not** a proof of
physical packet occupancy including NIC cell accounting, pending ACKs and
retransmissions. In particular, restoring early credit release invalidates
using the former conservative combined ring/arena packet envelope as an
active-session guarantee. No such guarantee is claimed here.

The conservative helpers remain unused by the active subset runtime and
planner: `subset.payload_budget`, and `routes.ring_window` /
`routes.forward_windows` with `subset_ring=True`. Their tests still verify
`4B + C(q) <= Q`, with `B = min(floor(.75Q), floor((Q-Cmax)/4))` and
`C(q) = n(q)*64*2*128`, and exercise bounded native ring progress.
That conditional model assumes MTU >= 256, at most 128 bytes of framing
and queue accounting per packet, no retained retransmission copies and
credit release only on returning completion. It applies only to those
explicitly bounded helper configurations, not the restored OFF pieces path.

## Larger-window analysis

The offline helper `subset.ring_packet_bound` fixes the conservative arena
windows from `forward_windows(..., chunk=4096, subset_ring=True)` and solves
each queue separately for a common ring window R. It does not configure the
runtime. For each directed QP i, let A_i be its arena window. Its shared
native `unacked` counter bounds arena plus ring payload by `max(A_i, R)`
when it carries the ring, and by A_i otherwise. Let D(q,R) sum those bounds
for every QP traversing q. The inequality is:

```text
1.5*D(q,R) + 0.5*D(reverse(q),R) + C(q) <= 524288
```

This uses `_roce_proxy.c`'s `post_stream_write`, `link_pump_windowed` and
`ack_push`: both pumps test the same device/peer byte counter. With delivery
proof disabled, only returning completion releases those bytes. Payload on
other QPs can remain live. The helper includes ACK-only queues and finds the
largest common 32 KiB-aligned window satisfying every inequality. This is the
maximum **within this fixed-arena, shared-QP envelope**. Trading arena capacity
for ring capacity changes the optimization problem. Control credits are also
shared by arena and ring on each native QP and are counted once.

For TP6 on positions 0-5 of ring8, the closing route is 5-6-7-0. A deployment
confined to 6-7 sends both its payload and ACKs over their direct cable, using
zero hairpin bytes. The limiting queues are the secondary function toward
port 0 on positions 6 and 7:

| Quantity | Bytes |
|---|---:|
| Conservative arena windows in each direction (three lanes) | 86,016 |
| Control reserve (six directed QPs times 16,384) | 98,304 |
| Arena window on the QP that also carries the ring | 28,672 |
| Other forward arena QPs, still independently live | 57,344 |
| Fixed other-QP framing, reverse arena ACKs and controls | 227,328 |
| Residual for shared ring-QP payload and framing | 296,960 |
| Unrounded ring payload allowance, floor(296960 / 1.5) | 197,973 |
| Largest common 32 KiB-aligned window | **196,608** |
| Occupancy bound at that window | 522,240 |
| Occupancy at the next 32 KiB window (229,376) | 571,392: refused |

This improves the retained helper's 65,536-byte common budget by three times.
It is not close to the normal 393,216-byte ring window. Even an isolated
393,216-byte data window would require 589,824 bytes with the minimum-MTU
framing assumption, before controls or reverse traffic. An isolated single
ring lane with no arena traffic can fit 327,680 bytes under this envelope;
that requires an arena/ACK drain, which the active subset runtime does not
establish. Larger MTU assumptions require negotiated-MTU enforcement and a
separate reviewed proof; measured absence of drops does not supply that proof.

The active arena windows are larger than the conservative helper's and retain
delivery-proof release. Its proof assumptions therefore do not hold merely by
setting `SIRCL_SUBSET_RING=1`. Subset ring all-reduce stays disabled, and flat
all-gather keeps chain. No default or opt-in transport setting changes here.

Using the inferred SIRCL window/RTT model in
[schedule selection](../../docs/architecture/sircl-schedule-selection.md),
196,608 bytes over two relays predicts 16.90 GB/s before kernel and cable/NIC
contention, or about 1.47 times 11.5 GB/s pieces. Applying the measured
57/46.4 RTT-growth sensitivity gives 13.76 GB/s, about 1.20 times pieces.
These are conditional steady-state predictions, not measured gains. A
co-resident DCP2 workload can further reduce available bandwidth. Using the
profiled net-sircl 40.3 us directly instead predicts 9.76 GB/s (0.85 times
pieces); the different transport/profiling conditions are material.

## Validation gates

Run `python -m pytest spark_transport/sircl -q`. The subset tests cover
admission, direct chain lanes, active payload bounds, retained conservative
helper bounds, runtime schedule selection, ON/OFF pieces equality, native
progress and OFF byte hashes generated from `df86c93d`.
GPU emulation must cover exact eager and captured/replayed flat and padded
gathers and all-reduces, with delayed link progress and matching native
ACK-release settings. Emulation does not measure hairpin drops or serving
performance.

The offline selector tests cover every path/cycle size from two to eight,
representative subset and co-resident placements, odd cycles, ACK-only queues,
measurement precedence and rejection of unbounded candidates. The fixed-arena
ring bound is checked for every admitted subset, including failure of the
next larger common window. The selector's drained candidates do not change
runtime admission. Both hardware plans, including the restoration budget and
the 25-minute feasibility limit, are in the
[selection design](../../docs/architecture/sircl-schedule-selection.md#hardware-validation-plans).

Before promotion, coordinate a maximum 25-minute TP6 outage on positions
0-5, preserving DCP2 on 6-7 with no other client. Pre-stage artifacts and
budget restoration time before taking TP6 down. Measure three ABBA blocks
(12 arms) of large flat/row/padded gathers and all-reduces, plus small
messages, using identical configurations except the opt-in. Require exact
results, zero drops on all eight Sparks, DCP2 decode within predeclared A/A
noise, and every case at least 0.97x OFF throughput (OFF/ON latency for
small messages). An incomplete block, noisy baseline or missed deadline
is inconclusive; restore TP6 within the reserved time and keep the mode off.
