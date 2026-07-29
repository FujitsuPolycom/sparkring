# SparkCache diagonal buddy replication

This directory defines the GPU-free protocol and bounded state machines for
replicating SparkCache chunks over the two otherwise-unused diagonal 10 GbE
links:

```text
rank 0 <========== 10 GbE ==========> rank 2
rank 1 <========== 10 GbE ==========> rank 3
```

It does **not** open sockets, bind NICs, change routes, or touch CUDA. TCP can
carry the protocol first; raw Ethernet or another carrier can replace it later
without changing cache transaction semantics.

## Transaction

```text
BEGIN(txid, generation, context, identity)
PUT_CHUNK(index, sha256, bytes) ...
COMMIT(sha256(commit-record), ordered chunk digests)

or:

ABORT(txid, generation)
```

The receiver calls its publication callback only after every ordered chunk and
the canonical commit record validate. Chunks staged before COMMIT are not a
visible replica. Duplicate BEGIN, PUT_CHUNK, COMMIT, and ABORT frames are
idempotent. A newer generation supersedes an abandoned older generation; stale
traffic is rejected.

ACK identifies the exact input sequence it acknowledges. CREDIT advertises
**absolute** free byte and frame capacity, so a replayed CREDIT cannot inflate
the sender window. Input sequences are scoped by transaction and generation,
so concurrent publications may each begin at sequence zero. Retransmission
reuses the exact encoded bytes and consumes no additional credit.

## Fail-open serving rule

Buddy replication is opportunistic. If receiver credit, sender window, remote
validation, or the publication callback fails, `BuddySender` marks only that
transaction local-only. It returns `None` instead of waiting for capacity and
queues one best-effort ABORT control frame to release any remote staging.
Inference and the local SparkCache commit remain outside this state machine.

The receiver independently bounds:

- active transactions;
- staged bytes and chunks that have not reached durable storage;
- per-context chunk descriptors;
- remembered commit receipts.

On a bound or integrity failure it drops the incomplete remote transaction and
returns credits. The receiver also expires inactive transactions after a
bounded TTL; a production carrier must call that expiry from its timer loop
even when no new frames arrive, then reconnect with a new generation after
connection loss.

Production mode supplies an `on_chunk` callback that durably publishes each
content-addressed chunk immediately. The receiver then retains only its digest
and index until COMMIT, so a 393K context's 1,535 rank-local chunks do not
accumulate 3.14 GB in RAM. The GPU-free gate exercises that full descriptor
count with a one-byte staging allowance.

## Tests

From the repository root:

```powershell
python -m pytest sparkcache/replication -q
```

The tests cover fragmented framing, canonical headers, size bounds,
manifest-last visibility, SHA-256 validation, duplicate delivery, stale
generations, aborts, credits, retransmission, and local-only degradation.
