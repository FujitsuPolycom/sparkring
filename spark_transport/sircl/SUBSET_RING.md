# Subset chain and ring sessions

**Experimental.** Set `SIRCL_SUBSET_RING=1` on the tensor-parallel session,
with `SIRCL_ROUTE_SCOPE=fabric` in the vLLM adapter. The serve and bundle
commands expose `--subset-ring`, which sets both variables. Choose
`--large-schedule ring --gather-schedule ring` to request ring schedules.
The setting is off by default. It does not change the fabric's relay rules.

Admission is limited to a cable-connected arc of at least three ranks on a
cycle of four to eight Sparks, leaving one or two adjacent non-members.
Wrapping arcs and reordered ranks are supported. Larger complements are
refused, even if idle: they can carry independent traffic through the same
queues and need a reservation protocol that this implementation does not
provide. This is not general support for every contiguous subset.

Use the universal, same-function relay table of the complete cycle. Lane
selection remains the universal shortest-path rule, including opposite-pair
ties. Setup checks each lane's physical port and function against that rule.
Extending a closing edge around the long side of a cycle would require relay
routes that the universal table does not necessarily carry; this setting
does not do so. Chain edges remain direct cables. Only the ring closes the
chain through relays.

## Relay capacity model

The queue key is `(relay position, function class, egress port)`. Host-originated
traffic does not enter this ingress-to-egress hairpin queue. For TP6 on
positions 0–5, the closing lanes from 5 to 0 pass through 6 and 7. TP2 and DCP2
traffic confined to 6–7 uses their direct cable in both directions, including
ACKs. Its contribution to every hairpin queue is zero, regardless of the
number of sessions, message size or forward-window setting. It can still
compete for egress bandwidth, so queue safety does not establish decode
performance isolation.

The following is a conservative **inferred packet envelope**, not a hardware
measurement. Its assumptions are a queue of 524,288 bytes, RC path MTU at
least 256 bytes, at most 128 bytes of framing and queue accounting per packet,
and no additional copy retained by retransmission. The driver and NIC's
internal cell accounting are not established by CPU or GPU emulation. Confirm
those assumptions and the drop counters before treating the bound as a
physical queue guarantee.

For queue `q`, let `n(q)` count all directed session lanes through `q` and its
reverse queue. The native sender permits at most 64 completion credits per
QP; allow two short packets per credit. Reserve
`C(q) = n(q) × 64 × 2 × 128` bytes for flags, credits, final partial packets
and their ACKs. With 128 bytes of overhead per 256 payload bytes, framing is
at most half the payload plus the partial-packet allowance. Reverse ACKs
are bounded the same way. A control write and its ACK belong to opposite
directions; both directions' lane counts appear in `n(q)`.

Let `Cmax` be the largest `C(q)` over the session's queues. Each operation
family gets a per-queue payload budget

```
B = min(floor(0.75 × Q), floor((Q − Cmax) / 4))
```

Arena traffic has at most `B` payload bytes per queue. Ring traffic has at
most `B` per queue. Allow both families to overlap, with the same allowances
in the reverse direction. Payload, framing and reverse ACKs then occupy at
most `2B + B + B + C(q) = 4B + C(q) ≤ Q`. No external relay share is assumed.
The admitted non-members contribute zero hairpin bytes by topology.

For the six-of-eight layout with two lanes, `Cmax = 163,840`, so `B = 90,112`.
Ring windows round down to 65,536 bytes (two 32-KiB chunks), instead of the
legacy 393,216-byte window. The two ring links share the QP's unacknowledged
byte counter; their windows do not add. Arena windows divide `B` among the
lanes sharing each queue and round down to 4-KiB chunks. The conservative
envelope on relays 6 and 7 is:

| Function and egress | Arena payload | Ring payload | Control allowance | Total including reverse ACKs |
|---|---:|---:|---:|---:|
| Primary, port 0 | 90,112 | 65,536 | 32,768 | 311,296 |
| Primary, port 1 | 90,112 | 0 | 32,768 | 245,760 |
| Secondary, port 0 | 86,016 | 65,536 | 98,304 | 368,640 |
| Secondary, port 1 | 86,016 | 0 | 98,304 | 303,104 |

All values are bytes per queue, and each row applies independently on both
relays. These are saturation bounds, not expected occupancies.

## Enforcement and progress

[subset.py](sparkring_sircl/subset.py) owns admission and the budget.
[routes.py](sparkring_sircl/routes.py) enforces the budget for ring and arena
windows. A disabled forward window, oversized queue declaration, non-universal
lane, or chunk that cannot fit its share is refused. A window is never rounded
up past its allowance. The 4-KiB forward chunk is a subset default; legacy
sessions keep their existing settings.

[runtime.py](sparkring_sircl/oneshot/runtime.py) admits the chain and link
areas, configures the bounded windows before starting the progress thread,
and disables delivery-proof credit release in the native constructor. Bytes
therefore stay charged until their RC completion, including the returning
ACK. A rank mismatch is rejected by setup agreement. The built-in tuning
table still requires full fabric coverage; no full-cycle measurement is
implicitly applied to a subset.

[budget.py](sparkring_sircl/p2p/budget.py) reserves the complete collective
share of every used relay queue, preventing additional relayed P2P QPs in
the same deployment. Direct P2P channels remain available. DCP2 sessions on
adjacent member pairs and on the two non-members use no hairpin queues; DCP
sessions do not inherit the tensor-parallel subset flag. The proof does not
cover another overlapping deployment on the member positions.

The native link pump polls completions and releases credits while payload
posting waits for window space. Every admitted ring has at least one chunk
of credit. Existing slot/stagger and thread-count checks remain in force.
These invariants remove a zero-credit cycle introduced by budgeting; they
are not a proof of progress after a failed rank or transport fault. The
native simulator tests repeated all-reduce and all-gather operations with a
delayed rank. CUDA graph progress also requires all emulated ranks' blocks
to be resident; production uses one GPU per rank.

## Harness comparison

The harness accepts `--session-env SIRCL_ROUTE_SCOPE=fabric` to keep the same
universal routes in both arms. An operator can plan the comparison offline:

```bash
python -m sparkring_sircl.ring plan --site "$SITE" --groups 0-5 --name subset6 \
  --large --large-sizes 1048576,8388608,33554432,67108864 \
  --chain-gather-sizes 131072,1048576,8388608 \
  --large-schedules ring --gather-schedules ring \
  --session-env SIRCL_ROUTE_SCOPE=fabric --session-env SIRCL_SUBSET_RING=1
```

For the off arm, keep `SIRCL_ROUTE_SCOPE=fabric`, set `SIRCL_SUBSET_RING=0`,
and select `pieces` for both schedules. Capture counters on **all eight**
Sparks independently: the six-rank harness does not collect the two
non-members' counters. Run only after a coordinating operator has arranged
the concurrent DCP2 workload and a relay-drop watchdog. Do not promote the
setting on emulation results alone.

## Compatibility tests

[test_subset_ring.py](tests/test_subset_ring.py) enumerates every subset of
size 3–7 on cycles of size 4–8, with one and two lanes and with the complement
active or idle. Admitted layouts must satisfy the packet envelope; others
must fail closed. This does not claim ring execution for rejected layouts.
Nine admitted size combinations run the production C link pump through the
CPU verbs stand-in. The stand-in does not model physical hairpin occupancy.

The off-setting fixture records ordered JSON hashes from commit `df86c93d`
for 20 layouts, four schedule requests per layout, setup records, lane maps,
forward windows and six complete bundle plans. Both an unset flag and `0`
must match those bytes. Build provenance is fixed for that comparison:
source digests necessarily change when implementation files change and must
not be falsified to preserve a receipt's build hash.
