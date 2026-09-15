# Profile validation report template

Copy this template into a named evidence record and replace bracketed fields.
Use **Development**, **Validated**, **Experimental**, or **Unsupported** for
the reader-facing profile status. Machine-readable status values follow the
[writing and evidence policy](development/writing.md). Record a gate as pass,
fail, pending, or not applicable. C1/C8 denote one/eight concurrent requests.

## Conditions

- Profile status: [Development | Validated | Experimental | Unsupported]
- Profile/recipe and source revision: [identifiers]
- Image digest; target/draft revisions; quantization: [identifiers]
- Hardware/topology, TP/DCP/PP: [values]
- Context, sequence, batch, KV budgets: [values]
- Kernels, graph sizes, transport routes, speculation: [values]
- SparkCache mode/namespace and initial cache state: [values]
- Benchmark revision; temperature/top-p/seeds; thinking/template: [values]
- Agreed performance and accuracy budgets: [values and reference profile]

## Measurement

Follow [the validation runbook](operations/profile-validation.md). Link commands and raw
receipts with hashes. Record timing definitions, actual token counts,
repetitions, concurrency, and cache evidence.
Use the selected profile's deployment guide for its startup and warmup gate.

## Result

| Check | Outcome | Numbers / evidence |
|---|---|---|
| Startup and warmup | [gate] | [all-rank readiness] |
| Prefill ×3 | [gate] | [per-context median/min/max TTFT and tokens/s] |
| Decode matrix ×3 | [gate] | [per-cell median/min/max throughput, TTFT, ITL] |
| Coding peak ×3 | [gate] | [median/peak rate, output lengths, truncation] |
| Long-context accuracy (Estonia), C1/C8 | [gate] | [correct/attempted and output-budget hits] |
| Needle/revision/join | [gate] | [actual contexts, positions, pass counts] |
| SparkCache restart restore ×3 | [gate or not applicable] | [per-rank restore and accuracy] |
| Mixed traffic ×3 | [gate] | [idle versus loaded throughput/ITL/TTFT] |
| 30-minute sustained run | [gate] | [errors, memory/cache trend, health] |
| Application-specific checks | [gate or not applicable] | [coding/schema/tools/multimodal] |

## Conclusion

[Summarize the measured strengths, any failed gates, and the intended workload.]

Operator recommendation decision: [retain / revise / recommend, with reason].
The report itself does not change the selected deployment or public defaults.

## Limitations

[State the tested workload boundaries and any required cells not exercised.]
