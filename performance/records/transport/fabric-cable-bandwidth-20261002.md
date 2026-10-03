# Fabric cable bandwidth after ring setup, 2026-10-02

Status: **research-only**. One four-Spark ring, one session of manual
measurements. This record is the basis of `sudo sparkring cabling --bandwidth`
([cable speed](../../../docs/operations/install-reference.md#cable-speed)).

## Question

Can a ConnectX fabric cable between two DGX Sparks stay up, count no errors,
and still carry much less than its normal bandwidth? Which measurement shows
that, what does it cost a model, and what restores the cable?

## Conditions

- Four DGX Sparks cabled as a SparkRing ring: each cable joins port 0 of
  rank r to port 1 of rank r+1. Each port appears as two PCIe functions that
  share its cable: port 0 as `enp1s0f0np0` (RDMA device `rocep1s0f0`) and
  `enP2p1s0f0np0` (`roceP2p1s0f0`), port 1 as `enp1s0f1np1` (`rocep1s0f1`)
  and `enP2p1s0f1np1` (`roceP2p1s0f1`). Two PCIe Gen5 x4 links, `enp1s0`
  and `enP2p1s0`, each carry one function of each port (`f0` and `f1`); a
  function reaches about 109 Gb/s each way, the limit of its PCIe link.
- Two cable models: Amphenol NJAAKK-N911 and Central CAB-200GDAC1.
- RoCE v2 GID index 3 of every function held its fabric IPv4 address, as
  SparkRing keeps it.
- Measured after ring setup work on the Sparks.
- The perftest version was not recorded.

## Measurement

Between the two ends of one function of one cable, with perftest's
`ib_write_bw`:

- server end: `ib_write_bw -b -d DEVICE -x 3 -s 1048576 -D 5 -F --report_gbits -p PORT`;
- client end: the same arguments followed by the server's fabric IPv4
  address on that function.

`-b` sends RDMA writes in both directions at once; `-x 3` selects RoCE GID
index 3. The figure is the client's `BW average[Gb/sec]` column, for
example this line of a healthy run:

```text
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 1048576    38096            0.00               213.05 		   0.025397
```

Functions and cables were measured one at a time: the two functions on one
PCIe link (`f0` and `f1` of `enp1s0`, or of `enP2p1s0`) share its bandwidth,
so concurrent tests disturb each other. One-way runs omitted `-b`. Errors were
read from the RoCE counters `out_of_sequence`, `packet_seq_err` and
`local_ack_timeout_err` and from the FEC counters.

## Results

| Cable state | Bidirectional, per function | One way, per function |
|---|---|---|
| Healthy | about 213 Gb/s | about 109 Gb/s |
| Degraded, three of the four ring cables | about 119 Gb/s (for example 118.66) | about 109 Gb/s |
| Degraded, one ring cable | about 26 Gb/s | not recorded |

- The degraded cables counted no RoCE retransmits (all three counters 0)
  and no uncorrectable FEC.
- FEC corrected bits do not separate healthy from degraded cables: healthy
  Amphenol NJAAKK-N911 cables logged about 1,000 corrected bits per 5 s at
  full speed, healthy Central CAB-200GDAC1 cables logged none.
- Prefill throughput of a GLM-5.3-Flash two-Spark (TP2) deployment, before
  and after the degraded cables were restored:

  | Prompt length | Degraded cable (tok/s) | Restored cable (tok/s) |
  |---|---|---|
  | 8K | 1,339 | 2,132 |
  | 64K | 1,589 | 2,510 |
  | 128K | 1,642 | 2,488 |

  Decode throughput looked normal over the degraded cable.
- Repair attempts on degraded cables:

  | Action | Result |
  |---|---|
  | `ip link set down` and `up` of the function | still degraded |
  | `devlink dev reload pci/BDF action driver_reinit` | still degraded |
  | Reboot both Sparks on the cable | 213 Gb/s |
  | Reboot one end of the 26 Gb/s cable | 119 Gb/s |

- On healthy cables, each of these left the bidirectional result at
  213 Gb/s: a driver reinit of a function; 2.7 TB of TCP over a cable;
  starting the four-Spark mesh; four-Spark (TP4) prefill load through the
  mesh; switching the ring between one four-Spark model and two two-Spark
  models on its halves, including the mesh stop and the RoCE GID restore;
  installing models on the halves.

## Conclusion

A cable can lose most of its bidirectional bandwidth (down to about 26 of
213 Gb/s per function) with no error counted, while a one-way test still
reaches the PCIe limit, so a bidirectional test per function detects it and
a one-way test does not. A 190 Gb/s threshold separates
the healthy value (about 213 Gb/s) from both degraded values with room for
run-to-run variation. On this ring the degraded state cost a two-Spark
model 34-37% of its prefill throughput. Rebooting both Sparks on the
cable restored it; a link reset or a driver reinit did not. None of the
SparkRing operations retested on healthy cables reproduced the state.

## Limitations

- One ring and one session; the number of degraded cables and their values
  apply to these Sparks and cables only.
- The cause is not established. Plugging cables in while the Sparks ran is
  the likely cause; it was not reproduced.
- The prefill benchmark's harness, request shape and repetition count are
  not part of this record; the throughput figures show the size of the
  effect, not a qualified result.
- A pair's cable between its ports 1 has no fabric addresses and was not
  measured this way.
