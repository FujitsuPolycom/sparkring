"""Server-side views of passive status facts, shared by HTML and text routes."""

from __future__ import annotations

from html import escape
from datetime import datetime, timezone
import json
import math
import re

from .collector import ARG_FIELDS


# label, configured argument or environment variable, resolved field.
GROUPS = {
    "Model and topology": [
        ("Served model name", None, "served_model_name"),
        ("Model architecture", None, "model_type"),
        ("Tensor parallel", "tensor_parallel_size", "tensor_parallel_size"),
        ("Decode context parallel", "decode_context_parallel_size", "decode_context_parallel_size"),
        ("Pipeline parallel", "pipeline_parallel_size", "pipeline_parallel_size"),
        ("Context limit", "max_model_len", "max_model_len"),
        ("Max sequences per batch", "max_num_seqs", "max_num_seqs"),
        ("Quantization", None, "quantization"), ("Model dtype", None, "model_dtype"),
    ],
    "Chat and tools": [
        ("Reasoning parser", "reasoning_parser", "reasoning_parser"),
        ("Tool-call parser", "tool_call_parser", None),
        ("Default chat template arguments", "default_chat_template_kwargs", None),
    ],
    "Prefill and attention": [
        ("Chunked prefill", "chunked_prefill_enabled", "chunked_prefill_enabled"),
        ("Max tokens per batch", "max_num_batched_tokens", "max_num_batched_tokens"),
        ("Activation quant fusion", None, "fuse_act_quant"),
        ("Attention backend", None, "attention_backend"),
        ("Attention layer types", None, "decoder_attention_implementations"),
        ("Q/K head dimensions", None, "qk_head_dims"),
        ("Value head dimensions", None, "value_head_dims"),
    ],
    "Decode and speculation": [
        ("Speculation method", None, "speculative_method"),
        ("Draft tokens", None, "num_speculative_tokens"),
        ("Draft model type", None, "draft_model_type"),
        ("Draft tensor parallel", None, "draft_tensor_parallel_size"),
        ("Draft load format", None, "draft_load_format"),
        ("Draft KV cache dtype", None, "draft_kv_cache_dtype"),
        ("Draft sampling", None, "draft_sample_method"),
        ("Rejection sampling", None, "rejection_sample_method"),
        ("Adaptive verification", None, "adaptive_verification"),
        ("CUDA graph mode", None, "cudagraph_mode"),
        ("Largest CUDA graph size", None, "max_cudagraph_capture_size"),
        ("Linear backend", None, "linear_backend"), ("MoE backend", None, "moe_backend"),
        ("MXFP8 LM head", "VLLM_MXFP8_LM_HEAD", None),
        ("LM head A16", "VLLM_LM_HEAD_A16", None),
    ],
    "Transport": [
        ('RoCEnante available (TP)', 'VLLM_ENABLE_ROCE_ALLREDUCE', 'tp_rocenante_enabled'),
        ('RoCE all-reduce size limit', 'VLLM_ROCE_ALLREDUCE_MAX_SIZE', 'tp_roce_allreduce_max_bytes'),
        ('RoCE all-gather shard size limit', 'VLLM_ROCE_ALLGATHER_MAX_SIZE', 'tp_roce_allgather_max_bytes'),
        ('NCCL runtime version (TP)', None, 'tp_nccl_version'),
        ('NCCL library name (TP)', None, 'tp_nccl_library_path'),
        ('RoCEnante selected HCAs', 'B12X_ROCE_HCA', 'tp_roce_hcas'),
        ('RoCEnante GID index', None, 'tp_roce_gid_index'),
        ('RoCEnante PCI domains', None, 'tp_roce_pci_domains'),
        ('PCI-domain routing preference', 'NCCL_IB_PRESERVE_PCI_DOMAIN', None),
        ('NCCL subnet-routing override', 'NCCL_IB_SUBNET_AWARE_ROUTING', None),
        ('NCCL algorithm override', 'NCCL_ALGO', None),
        ('NCCL protocol override', 'NCCL_PROTO', None),
        ('NCCL minimum channels', 'NCCL_MIN_NCHANNELS', None),
        ('NCCL maximum channels', 'NCCL_MAX_NCHANNELS', None),
        ('SIRCL ring sessions', 'SIRCL_MODE', None),
        ('SIRCL NCCL setting', 'SIRCL_NCCL', None),
        ('SIRCL fabric', 'SIRCL_FABRIC', None),
        ('SIRCL fabric positions', 'SIRCL_RANK_POSITIONS', None),
        ('SIRCL session (TP)', None, 'tp_sircl_session'),
        ('SIRCL session state (TP)', None, 'tp_sircl_state'),
        ('SIRCL NCCL policy (TP)', None, 'tp_sircl_nccl'),
        ('SIRCL PyNccl (TP)', None, 'tp_sircl_pynccl'),
        ('SIRCL relays on a lane (TP)', None, 'tp_sircl_relays'),
        ('SIRCL receipt age (TP)', None, 'tp_sircl_receipt_age_s'),
    ],
    "Cache and loader": [
        ("KV transfer", None, "kv_transfer_enabled"),
        ("KV connector", None, "kv_connector"),
        ("GPU prefix cache", "prefix_caching_enabled", "prefix_caching_enabled"),
        ("KV cache dtype", None, "cache_dtype"),
        ("KV cache memory per worker", None, "kv_cache_memory_bytes"),
        ("KV cache block size", "block_size", "block_size"),
        ("Mamba cache mode", None, "mamba_cache_mode"),
        ("Recurrent checkpoint policy", None, "recurrent_checkpoint_policy"),
        ("Load format", "load_format", "load_format"),
        ("Loader read mode", None, "loader_read_mode"), ("Loader I/O threads", None, "loader_io_threads"),
    ],
}
QWEN = [
    ("Hyper-connection prefill row ownership", "VLLM_QWEN3_8_HC_PREFILL_MODE", "hc_prefill_row_ownership"),
    ("Hyper-connection projection ranks", None, "hc_projection_tp_size"),
    ("Qwen prefill coalescing", "VLLM_QWEN3_8_PREFILL_COALESCE", None),
    ("Compact MTP", "VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT", None),
    ("Projection overlap", "VLLM_QWEN3_8_FLASH_NEXT_OVERLAP", None),
    ("MTP NVFP4 head", "VLLM_MTP_NVFP4_LM_HEAD", None),
    ("MoE input scaling", "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE", None),
    ("GDN speculative-decode metadata fast path", "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH", None),
]
MIMO = [
    ("FP4 MoE A16", "VLLM_B12X_MOE_FP4_FORCE_A16", None),
    ("Auxiliary MXFP8 streaming", "VLLM_DFLASH_AUX_MXFP8_STREAMING", None),
    ("Auxiliary BF16 staging", "VLLM_DFLASH_AUX_BF16_STAGING", None),
    ("Compact RoPE", "VLLM_DFLASH_COMPACT_ROPE", None),
    ("Sharded auxiliary projection", "VLLM_DFLASH_SHARD_AUX_PROJECTION", None),
]
NOTE = "Unknown values do not mean OFF."
# Settings that vLLM changes from the configured value by design. The runtime
# value carries a footnote marker instead of a warning.
ADJUSTED_BY_VLLM = {
    'block_size': 'vLLM adjusts the KV cache block size to suit the model, '
                  'for example enlarging it to fit the model\'s recurrent state.',
}
# Plain names for fact sources and reasons, shown in the Source tooltip.
SOURCE_NAMES = {'resolved_vllm_config': 'vLLM configuration in the API process',
                'parsed_server_arguments': 'Launch arguments',
                'process_environment': 'Environment variables',
                'worker_instance': 'Worker process'}
REASON_NAMES = {'not_set_in_environment': 'Not set in environment', 'invalid_value': 'Invalid value',
                'not_supplied': 'Not in launch arguments', 'per_collective_or_not_exposed': 'Not exposed by NCCL',
                'not_collected': 'Not collected', 'not_applicable': 'Not applicable',
                'unsupported_value': 'Value not shown'}
# Plain names for the JSON workers.state values.
STATE_NAMES = {'complete': 'Complete', 'partial': 'Incomplete', 'pending': 'Refreshing',
               'error': 'Worker request failed', 'unavailable': 'Workers unavailable',
               'initializing': 'Starting up'}
ENV_RESOLVED = {
    'VLLM_ENABLE_ROCE_ALLREDUCE': 'tp_rocenante_enabled',
    'VLLM_ROCE_ALLREDUCE_MAX_SIZE': 'tp_roce_allreduce_max_bytes',
    'VLLM_ROCE_ALLGATHER_MAX_SIZE': 'tp_roce_allgather_max_bytes',
    'VLLM_MXFP8_LM_HEAD': 'target_head_mxfp8',
    'VLLM_LM_HEAD_A16': 'target_head_a16',
    'VLLM_MTP_NVFP4_LM_HEAD': 'draft_head_nvfp4',
}
LOCAL_FIELDS = {'tp_roce_hcas', 'tp_roce_gid_index', 'tp_roce_pci_domains', 'tp_nccl_library_path',
                'tp_sircl_session', 'tp_sircl_receipt_age_s'}
AUTO_FIELDS = {'model_dtype', 'quantization', 'attention_backend', 'draft_tensor_parallel_size',
               'draft_kv_cache_dtype', 'draft_sample_method', 'rejection_sample_method',
               'cudagraph_mode', 'gdn_prefill_backend', 'max_cudagraph_capture_size',
               'fuse_act_quant', 'max_model_len', 'max_num_seqs', 'max_num_batched_tokens',
               'block_size', 'kv_cache_memory_bytes', 'prefix_caching_enabled', 'chunked_prefill_enabled',
               'draft_load_format', 'linear_backend', 'moe_backend'}
RUNTIME_ONLY = {'served_model_name': 'From launch arguments',
                'model_type': 'From checkpoint', 'decoder_attention_modules': 'Read from the model',
                'sampled_qk_head_dims': 'From model', 'sampled_value_head_dims': 'From model',
                'hc_projection_tp_size': 'From topology', 'target_head_quantization': 'Chosen at model load',
                'draft_head_quantization': 'Chosen at model load', 'draft_head_shared': 'Chosen at model load'}
GROUPS['Prefill and attention'] = [
    (label, cfg, {'decoder_attention_implementations': 'decoder_attention_modules',
                  'qk_head_dims': 'sampled_qk_head_dims', 'value_head_dims': 'sampled_value_head_dims'}.get(actual, actual))
    for label, cfg, actual in GROUPS['Prefill and attention']]
GROUPS['Prefill and attention'].append(('GDN prefill backend', 'gdn_prefill_backend', 'gdn_prefill_backend'))
GROUPS['Decode and speculation'] += [
    ('Target head precision', None, 'target_head_quantization'),
    ('Draft head precision', None, 'draft_head_quantization'),
    ('Draft shares target head', None, 'draft_head_shared'),
    ('Shared-memory reader window', 'SPARKRING_SHM_BUSY_LOOP_S', None),
]
# Launch arguments that only the API server reads. Workers have no copy, so
# their rows show no per-worker comparison.
API_SERVER_ONLY = {'tool_call_parser', 'default_chat_template_kwargs'}
GROUP_NOTES = {
    'Chat and tools': 'Each request can set thinking with chat_template_kwargs or reasoning_effort. '
                      'Request values take precedence over these defaults.',
}
SAVE_CPU_WINDOW_S = 0.002  # runtime/common/serving.py SWITCHES["save_cpu"]


def shm_window(record):
    """The shared-memory reader window, named by where its value comes from."""
    if record.get('reason') == 'not_set_in_environment':
        return '1 s (vLLM default)'
    if known(record) and record['value'] == SAVE_CPU_WINDOW_S:
        return '2 ms (sparkring install --save-cpu)'
    if known(record) and type(record['value']) in (int, float):
        return f"{record['value']} s"
    return None


def template_defaults(record):
    """Unset defaults leave thinking and similar options to the model's chat template."""
    if known(record) and record['value'] is None:
        return 'Not set (the chat template decides)'
    return None


# Settings whose values read better with a fixed phrase; None falls back to display().
PHRASES = {'SPARKRING_SHM_BUSY_LOOP_S': shm_window, 'default_chat_template_kwargs': template_defaults}


def clean(value):
    """Prevent terminal control sequences in values and keep one fact per line."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", "?", str(value))


def known(record):
    return isinstance(record, dict) and record.get("state") == "known" and "value" in record


def fact_value(record):
    return record["value"] if known(record) else None


def display(record, key=""):
    phrase = PHRASES[key](record) if key in PHRASES else None
    if phrase is not None:
        return phrase
    if not known(record):
        return REASON_NAMES.get(record.get('reason'), 'Unknown')
    value = record["value"]
    if value is None:
        if record.get('reason') == 'not_applicable':
            return 'Not applicable'
        return 'Automatic' if key in AUTO_FIELDS else 'None'
    if value == '':
        # vLLM stores an empty name for an unselected option, such as the reasoning parser.
        return 'None'
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if key.endswith(("_BYTES", "_MAX_SIZE", "_bytes")) and isinstance(value, (int, float)):
        for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
            if value >= size:
                return f"{value / size:g} {unit}"
    if isinstance(value, int):
        if key == 'tp_nccl_version':
            return f'{value // 10000}.{value % 10000 // 100}.{value % 100}'
        return f"{value:,}"
    if isinstance(value, list):
        return ", ".join(clean(item) for item in value) or "None"
    return clean(value)


def configured(doc, key):
    if key is None:
        return {}
    group = "environment" if key.isupper() else "arguments"
    return doc.get("configured", {}).get(group, {}).get(key, {})


def effective(doc, key):
    return doc.get("effective", {}).get(key, {}) if key else {}


def base_value(record):
    value = record.get('value')
    return value.removeprefix('torch.') if type(value) is str else value


def signature(record):
    return json.dumps(base_value(record), sort_keys=True)


def refines(general, specific, key):
    """Whether a dtype names a more specific form of another, such as fp8_ds_mla of fp8.

    vLLM workers can store the concrete cache format of a configured family, and
    auto resolves to a concrete dtype; neither is a disagreement.
    """
    general, specific = base_value(general), base_value(specific)
    return (key.endswith('_dtype') and type(general) is str and type(specific) is str and general != specific
            and (general == 'auto' or specific.startswith(general + '_')))


SEVERITY = ('neutral', 'warn', 'bad')


def worse(*levels):
    return max(levels, key=SEVERITY.index)


def workers_phrase(count):
    return 'the worker' if count == 1 else 'both workers' if count == 2 else f'all {count} workers'


def origin(source, reason):
    """Plain description of where a value came from, for the Source tooltip."""
    text = ('Read from the running worker' if source.startswith('resident_')
            else SOURCE_NAMES.get(source, source.replace('_', ' ').capitalize()))
    if reason:
        text += ('; ' if text else '') + REASON_NAMES.get(reason, reason.replace('_', ' ').capitalize())
    return text


def agreement_of(ranks, reports, available, distinct, expected, unique_ranks, local):
    """Describe how the workers' reports of one setting compare, and how serious a gap is."""
    count = len(available)
    if local and available:
        return f'Per node · {count} of {expected or "?"} reported', 'neutral'
    if len(distinct) > 1:
        return 'Workers report different values', 'bad'
    if available and expected and count == expected:
        if unique_ranks:
            return ('Reported by the worker' if count == 1 else 'Same on ' + workers_phrase(count)), 'neutral'
        return f'{count} of {expected} reported, but rank numbers are missing or repeated', 'warn'
    if available:
        return (f'Only {count} of {expected} workers reported' if expected
                else f'Reported by {count} worker{"s" if count != 1 else ""}; expected count unknown'), 'warn'
    unset = sum(value.get('reason') == 'not_set_in_environment' for _, value in reports)
    if unset:
        return 'Not set on ' + (workers_phrase(unset) if unset == expected
                                else f'{unset} of {expected or "?"} workers'), 'neutral'
    return ('Unknown on ' + workers_phrase(len(ranks)) if ranks else 'No worker reports'), 'neutral'


def summarize(doc):
    workers = doc.get("workers", {})
    ranks = workers.get("ranks", [])
    expected = workers.get("expected_count")
    unique_ranks = workers.get("rank_identities_unique") is True
    identity = doc.get("provenance", {}).get("startup_identity", {})
    release = identity.get("sparkring")
    base = identity.get("eugr")
    title = "SparkRing" + (" " + clean(release) if release else " runtime status")
    if base:
        title += " · " + clean(base)
    groups = []
    for group, fields in GROUPS.items():
        fields = list(fields)
        model = str(fact_value(effective(doc, "model_type")) or "").lower()
        if group == "Prefill and attention" and "qwen" in model:
            fields += QWEN
        if group == "Decode and speculation" and "mimo" in model:
            fields += MIMO
        rows, group_notes = [], []
        for label, config_key, actual_key in fields:
            actual_key = actual_key or ENV_RESOLVED.get(config_key)
            config_key = config_key or (actual_key if actual_key in ARG_FIELDS else None)
            key = actual_key or config_key or ''
            requested = configured(doc, config_key)
            actual = effective(doc, actual_key)
            api_only = config_key in API_SERVER_ONLY and not actual_key
            reports = [] if api_only else [
                (r.get("identity", {}).get("rank", {}),
                 effective(r, actual_key) if actual_key else configured(r, config_key)) for r in ranks]
            available = [value for _, value in reports if known(value)]
            distinct = {signature(value) for value in available}
            agreement, agreement_severity = ('API server only', 'neutral') if api_only else agreement_of(
                ranks, reports, available, distinct, expected, unique_ranks, actual_key in LOCAL_FIELDS)
            if not known(actual) and len(distinct) == 1:
                actual = available[0] if actual_key else actual
            resolved_label = display(actual, key) if actual_key else 'Not checked at runtime'
            notes, notes_severity, footnote = [], 'neutral', None
            automatic = config_key in AUTO_FIELDS and known(requested) and requested['value'] in (None, 'auto')
            if (actual_key not in LOCAL_FIELDS and not automatic and known(requested) and known(actual)
                    and signature(requested) != signature(actual) and not refines(requested, actual, key)):
                if actual_key in ADJUSTED_BY_VLLM:
                    footnote = ADJUSTED_BY_VLLM[actual_key]
                else:
                    notes.append('vLLM changed the configured value')
                    notes_severity = 'warn'
            reference = actual if actual_key else requested
            if actual_key not in LOCAL_FIELDS and known(reference):
                differing = [value for value in available if signature(value) != signature(reference)]
                if differing and len(distinct) == 1 and refines(reference, differing[0], key):
                    resolved_label += f' ({display(differing[0], key)} on workers)'
                elif differing:
                    notes.append('The API and the workers report different values')
                    notes_severity = 'bad'
            if footnote:
                if footnote not in group_notes:
                    group_notes.append(footnote)
                marker = "*" * (group_notes.index(footnote) + 1)
                resolved_label += ' ' + marker
                footnote = marker + ' ' + footnote
            source = actual.get('source', '')
            # An unset variable whose phrase names the resulting default is still a launch setting.
            defaulted = config_key in PHRASES and requested.get('reason') == 'not_set_in_environment'
            evidence = (('Running worker' if source.startswith('resident_') else 'vLLM config') if known(actual)
                        else 'Launch setting only' if known(requested) or available or defaulted else 'Not reported')
            reason = actual.get('reason') or requested.get('reason')
            if actual_key in LOCAL_FIELDS and len(distinct) > 1:
                resolved_label = 'Differs by node'
            rows.append({"label": label, "configured": display(requested, config_key or "") if config_key else RUNTIME_ONLY.get(actual_key, 'No separate setting'),
                         "resolved": resolved_label, "footnote": footnote,
                         "evidence": evidence, "origin": origin(source or requested.get('source', ''), reason),
                         "agreement": agreement, "agreement_severity": agreement_severity,
                         "comparison": ". ".join(notes), "comparison_severity": notes_severity,
                         "severity": worse(agreement_severity, notes_severity),
                         'source': source, 'reason': reason,
                         "rank_values": [(display(rank), display(value, key)) for rank, value in reports]})
        groups.append((group, rows))
    state = clean(workers.get("state", doc.get("state", "unavailable")))
    stale = workers.get("stale") is True
    complete = state == "complete" and unique_ranks and not stale
    age = workers.get("cache_age_seconds")
    # The header names the model as clients request it. The architecture has its
    # own line and never substitutes for an unknown name, because an architecture
    # such as qwen4_exp is not a value clients can put in a request.
    return {"title": title, "identity": identity, "provenance": doc.get("provenance", {}),
            "model": display(effective(doc, "served_model_name")),
            "architecture": display(effective(doc, "model_type")),
            "topology": " / ".join(name + " " + display(effective(doc, key)) for name, key in (
                ("TP", "tensor_parallel_size"), ("DCP", "decode_context_parallel_size"), ("PP", "pipeline_parallel_size"))),
            "workers": f"{workers.get('received_count', 0)}/{expected if expected is not None else '?'}",
            "state": state, "state_label": STATE_NAMES.get(state, state), "stale": stale, "complete": complete,
            "age": f"{age:.2f}s" if isinstance(age, (int, float)) else "Not collected",
            "groups": groups, "ranks": ranks,
            "issue_count": sum(row["severity"] in ("warn", "bad") for _, rows in groups for row in rows)}


def footnotes(rows):
    return list(dict.fromkeys(row["footnote"] for row in rows if row["footnote"]))


def group_notes(group, rows):
    """Row footnotes, then the group's own note."""
    return footnotes(rows) + ([GROUP_NOTES[group]] if group in GROUP_NOTES else [])


def render_text(doc):
    view = summarize(doc)
    lines = [view["title"], f"Model: {view['model']} | {view['topology']}",
             f"Model architecture: {view['architecture']}",
             f"Workers reporting: {view['workers']} ({view['state_label'].lower()}"
             + ("; out of date" if view["stale"] else "") + f") | data age: {view['age']}", "", NOTE]
    for section in detail_sections(doc, view):
        lines += ["", section['title'].upper(), section['note'], " | ".join(section['headers'])]
        lines += [" | ".join(clean(value) for value in row) for row in section['rows']] or ['Not reported']
    for group, rows in view["groups"]:
        lines += ["", group.upper()]
        table = [("SETTING", "CONFIGURED", "RUNTIME VALUE", "SOURCE", "WORKERS")]
        table += [(r["label"], r["configured"], r["resolved"], r["evidence"],
                   r["agreement"] + ("; " + r["comparison"] if r["comparison"] else "")) for r in rows]
        widths = [max(len(row[i]) for row in table) for i in range(4)]
        lines += ["  ".join(value.ljust(widths[i]) if i < 4 else value for i, value in enumerate(row)) for row in table]
        for row in rows:
            if row["severity"] == "bad":
                lines.append("  " + row["label"] + ": " + "; ".join(f"rank {rank}={value}" for rank, value in row["rank_values"]))
        lines += group_notes(group, rows)
    lines += ["", "WORKERS", "Rank | Draft tokens | KV transfer | Kernel setup"]
    lines += [" | ".join(row) for row in rank_rows(view)] or ["No worker reports"]
    lines += ["", "BUILD INFORMATION"]
    lines += [f"{key}: {clean(value)}" for key, value in identity_items(view)]
    return ("\n".join(lines) + "\n").replace("—", "--").replace(" · ", " | ")


def rank_rows(view):
    result = []
    for rank in view["ranks"]:
        preparation = effective(rank, "kernel_preparation")
        state = preparation.get("session_state", preparation.get("state", "unknown"))
        if state in (None, "unknown"):
            state = "Unknown"
        if rank.get("error"):
            state = "Worker report failed"
        result.append((display(rank.get("identity", {}).get("rank", {})),
                       display(effective(rank, "num_speculative_tokens")),
                       display(effective(rank, "kv_transfer_enabled")), clean(state)))
    return result


def identity_items(view):
    identity = view["identity"]
    items = [(label, identity[key]) for label, key in (
        ("SparkRing", "sparkring"), ("Base image", "eugr"), ("Image digest", "image_reference"),
        ("vLLM package", "vllm_version"), ("B12X package", "b12x_version")) if identity.get(key)]
    provenance = view["provenance"]
    for component, source in provenance.get("sources", {}).items():
        if isinstance(source, dict) and source.get("commit"):
            items.append((component + " source commit", source["commit"]))
    if provenance.get("composition_sha256"):
        items.append(("Image composition SHA-256", provenance["composition_sha256"]))
    return items or [("Build information", "Not reported")]


def numeric(value, suffix='', digits=1):
    if type(value) not in (int, float) or not math.isfinite(value):
        return 'Unknown'
    return f'{value:,.{digits}f}{suffix}'


def byte_count(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return 'Unknown'
    for unit, size in (('TiB', 1024**4), ('GiB', 1024**3), ('MiB', 1024**2), ('KiB', 1024)):
        if value >= size:
            return numeric(value / size, ' ' + unit)
    return numeric(value, ' B', 0)


def collected_time(value):
    if type(value) is int and 0 <= value <= 2**63 - 1:
        return datetime.fromtimestamp(value / 1e9, timezone.utc).strftime('%H:%M:%S UTC')
    return 'Unknown'


def detail_sections(doc, view):
    """One table model for HTML and text; never infer measurements from settings."""
    sections = []

    def section(key, title, note, headers, rows, opened=True):
        sections.append({'id': key, 'title': title, 'note': note, 'headers': headers,
                         'rows': rows, 'open': opened})

    acceptance = doc.get('observed', {}).get('speculative_acceptance') or {}
    acceptance_rows = []
    reasons = {'needs_previous_snapshot': 'Waiting for the next refresh',
               'no_drafts_in_window': 'No drafts in this period', 'counter_reset': 'Counters restarted',
               'counters_unavailable': 'Counters unavailable',
               'counter_snapshot_unavailable': 'Counters unavailable',
               'inconsistent_counter_snapshot': 'Counter readings did not match'}
    for key, label in (('recent', 'Since previous refresh'), ('lifetime', 'Since the model started')):
        values = acceptance.get(key, {})
        if values.get('state') == 'known':
            rate = values.get('acceptance_rate')
            acceptance_rows.append((label, numeric(values.get('rounds'), digits=0),
                numeric(rate * 100, '%') if type(rate) in (int, float) else 'Unknown',
                numeric(values.get('estimated_tokens_per_verification'), digits=2),
                numeric(values.get('window_seconds'), ' s') if key == 'recent' else 'Since start'))
        else:
            reason = values.get('reason', acceptance.get('reason'))
            acceptance_rows.append((label, reasons.get(reason, 'No data'), 'No data', 'No data', 'No data'))
    section('acceptance', 'Draft acceptance',
        "From vLLM's speculative decoding counters. Tokens per step is an estimate: 1 + accepted draft "
        'tokens per verification step. It is not a speed in tokens per second. Acceptance depends on the '
        'prompts and sampling settings.',
        ('Period', 'Verification steps', 'Acceptance rate', 'Tokens per step', 'Duration'), acceptance_rows)
    positions = acceptance.get('lifetime', {}).get('positions', [])
    if positions:
        section('acceptance-positions', 'Acceptance by draft position', 'Totals since the model started.',
            ('Draft position', 'Drafted', 'Accepted', 'Acceptance rate'),
            [(str(row['position']), numeric(row['drafted'], digits=0), numeric(row['accepted'], digits=0),
              numeric(row['acceptance_rate'] * 100, '%')) for row in positions], False)

    memory_rows, disk_rows, version_rows, library_rows, package_rows = [], [], [], [], []
    group_rows, singleton_rows, nic_rows, address_rows, counter_rows = [], [], [], [], []
    for rank in view['ranks']:
        label = display(rank.get('identity', {}).get('rank', {}))
        resources = rank.get('resources') or {}
        memory = resources.get('memory') or {}
        container = resources.get('container_memory') or {}
        limit = ('Unlimited' if container.get('state') == 'known' and container.get('limit_bytes') is None
                 else byte_count(container.get('limit_bytes')))
        memory_rows.append((label, clean(resources.get('node', 'Not reported')),
            byte_count(memory.get('used_bytes')), byte_count(memory.get('available_bytes')),
            byte_count(memory.get('total_bytes')), byte_count(memory.get('swap_used_bytes')),
            byte_count(container.get('current_bytes')) + ' / ' + limit,
            collected_time(resources.get('collected_at_unix_ns'))))
        for disk in resources.get('filesystems', []):
            reason = disk.get('reason')
            scope = ('Network or unknown filesystem; not checked' if reason == 'remote_or_unrecognized_filesystem_not_probed_in_worker'
                     else 'Local filesystem' if disk.get('state') == 'known' else 'Not available')
            disk_rows.append((label, ', '.join(disk.get('paths', [])), byte_count(disk.get('available_bytes')),
                              byte_count(disk.get('used_bytes')), byte_count(disk.get('total_bytes')), scope))
        versions = rank.get('versions') or {}
        version_rows.append((label, display(versions.get('host_nvidia_driver', {})),
            display(versions.get('torch_build_cuda', {})), display(versions.get('cuda_toolkit', {})),
            display(versions.get('cuda_compiler', {})), display(effective(rank, 'tp_nccl_version'), 'tp_nccl_version')))
        for name, value in versions.get('packages', {}).items():
            package_rows.append((label, name, display(value)))
        for library in versions.get('mapped_libraries', []):
            library_rows.append((label, library.get('component', 'Unknown'),
                                 library.get('version') or 'Not in file name', library.get('path', 'Unknown')))
        transport = rank.get('transport') or {}
        for name, group in transport.get('groups', {}).items():
            if group.get('state') != 'known':
                continue
            roce, nccl = group.get('rocenante', {}), group.get('nccl', {})
            if (fact_value(group.get('world_size')) == 1
                    and fact_value(roce.get('enabled')) is not True
                    and fact_value(nccl.get('available')) is not True):
                singleton_rows.append((label, name, '1', 'Not needed (one worker)'))
                continue
            group_rows.append((label, name, display(group.get('world_size', {})),
                display(roce.get('enabled', {})), display(nccl.get('available', {})),
                display(roce.get('allreduce_max_bytes', {}), 'ar_bytes'),
                display(roce.get('allgather_max_bytes', {}), 'ag_bytes'),
                display(roce.get('hcas', {}))))
        for nic in transport.get('nics', []):
            pci = nic.get('pci', {})
            pcie = display(pci.get('current_link_speed', {})) + ' GT/s × ' + display(pci.get('current_link_width', {}))
            for net in nic.get('netdevs', []) or [{}]:
                speed = fact_value(net.get('speed_mbps'))
                nic_rows.append((label, nic.get('hca', 'Unknown'), display(pci.get('bdf', {})),
                    net.get('name', 'Unknown'), display(net.get('operstate', {})),
                    numeric(speed / 1000, ' Gb/s') if type(speed) in (int, float) else 'Unknown',
                    pcie, display(net.get('mtu', {}))))
                address_rows.append((label, net.get('name', 'Unknown'), display(net.get('mac', {})),
                    display(net.get('ip_addresses', {})), display(pci.get('device_key', {}))))
            for port in nic.get('ports', []):
                counters = port.get('counters', {})
                def traffic(name):
                    words = fact_value(counters.get(name))
                    return byte_count(words * 4) if type(words) is int else 'Unknown'
                counter_rows.append((label, nic.get('hca', 'Unknown'), str(port.get('port', '?')),
                    display(port.get('state', {})), traffic('port_xmit_data'), traffic('port_rcv_data'),
                    display(counters.get('port_rcv_errors', {})), display(counters.get('port_xmit_discards', {}))))

    section('nodes', 'Memory',
        'Used is total minus available. On a Spark, the CPU and GPU share this memory. '
        'Container memory is counted separately; do not add it to the host figures.',
        ('Rank', 'Hostname', 'Used', 'Available', 'Total', 'Swap used', 'Container used / limit', 'Read at'), memory_rows)
    section('storage', 'Disk space',
        'Available is the space a process without root rights can use. Paths on the same filesystem '
        'share one row. Network filesystems are not checked.',
        ('Rank', 'Paths', 'Available', 'Used', 'Total', 'Type'), disk_rows)
    section('versions', 'CUDA and NCCL versions',
        'The CUDA version PyTorch was built for, the installed CUDA toolkit and the loaded NCCL can differ. '
        'Installing a newer toolkit does not rebuild PyTorch or its extensions.',
        ('Rank', 'NVIDIA driver', 'PyTorch built for CUDA', 'CUDA toolkit', 'CUDA compiler (nvcc)', 'NCCL loaded for TP'), version_rows)
    section('libraries', 'Loaded libraries',
        'Libraries loaded by each worker process. The versions here come from file names; '
        'the NCCL version above comes from NCCL itself.',
        ('Rank', 'Library', 'Version in file name', 'Path'), library_rows, False)
    section('packages', 'Installed packages', 'Python package versions as installed. Source commits and loaded '
        'libraries are listed separately.',
        ('Rank', 'Package', 'Version'), package_rows, False)
    section('communicators', 'Communication groups',
        'What each group can use and its size limits. Available does not mean a request used it. '
        'NCCL does not report its algorithm, protocol or channel count here.',
        ('Rank', 'Group', 'Size', 'RoCEnante available', 'NCCL available', 'All-reduce limit', 'All-gather shard limit', 'Selected HCAs'), group_rows)
    if singleton_rows:
        section('singleton-groups', 'Single-worker groups',
            'These groups have one worker, so they need no network link. This is expected, not a network fault.',
            ('Rank', 'Group', 'Size', 'Network link'), singleton_rows, False)
    section('links', 'Network adapters and link rates',
        'Link rate is the speed the link negotiated, not measured throughput. The PCI domain is the first part '
        'of the PCI address. Interfaces with the same PCI device key share one physical link; do not add their rates.',
        ('Rank', 'HCA', 'PCI address', 'Interface', 'State', 'Link rate', 'PCIe link', 'MTU'), nic_rows)
    section('addresses', 'Network adapter addresses',
        'Only the adapters RoCEnante selected are listed. IP addresses are not collected.',
        ('Rank', 'Interface', 'MAC', 'IP addresses', 'PCI device key'), address_rows, False)
    section('rdma-counters', 'RDMA port traffic',
        'Running totals for the whole host, not only this model. This page runs no bandwidth or latency test.',
        ('Rank', 'HCA', 'Port', 'State', 'Transmitted', 'Received', 'Receive errors', 'Transmit discards'), counter_rows, False)
    return sections


def render_report(doc):
    """Escape every runtime value; only static markup enters the HTML response."""
    view = summarize(doc)
    e = lambda value: escape(clean(value), quote=True)
    state_class = "good" if view["complete"] else "warn"
    parts = [f'<main id="report"><header><p class="eyebrow">MODEL SERVER STATUS</p><h1>{e(view["title"])}</h1>',
             f'<p class="subtitle">{e(view["model"])} <span>·</span> {e(view["topology"])}</p>',
             f'<p class="subtitle">Model architecture: {e(view["architecture"])}</p></header>',
             '<div class="cards">',
             f'<section class="card"><span>Workers reporting</span><strong>{e(view["workers"])}</strong><small class="{state_class}">{e(view["state_label"])}{ " · out of date" if view["stale"] else ""}</small></section>',
             f'<section class="card"><span>Data age</span><strong>{e(view["age"])}</strong><small>Time since the workers last reported</small></section>',
             f'<section class="card"><span>Settings to check</span><strong>{view["issue_count"]}</strong><small>Values that differ, were changed, or are missing on some workers</small></section></div>',
             f'<p class="notice">{e(NOTE)}</p>']
    parts.append('<nav class="section-links" aria-label="Report sections"><a href="#group-0">Settings</a>'
                 '<a href="#nodes">Memory and disk</a><a href="#communicators">Transport</a>'
                 '<a href="#versions">Versions</a><a href="#acceptance">Draft acceptance</a></nav>')
    parts.append('<p class="legend">Configured shows launch arguments, including their defaults, and environment variables. '
                 'No separate setting means the value has no launch option of its own. '
                 'Not set in environment means the variable is absent; the value can still come from a default or a configuration file.</p>')
    for section in detail_sections(doc, view):
        opened = ' open' if section['open'] else ''
        parts += [f'<details id="{e(section["id"])}" class="group"{opened}><summary>{e(section["title"])}</summary>',
                  f'<p class="section-note">{e(section["note"])}</p>', '<div class="table-scroll"><table><thead><tr>',
                  ''.join(f'<th scope="col">{e(label)}</th>' for label in section['headers']), '</tr></thead><tbody>']
        for row in section['rows']:
            parts.append('<tr>' + ''.join(f'<td>{e(value)}</td>' for value in row) + '</tr>')
        if not section['rows']:
            parts.append(f'<tr><td colspan="{len(section["headers"])}">Not reported</td></tr>')
        parts.append('</tbody></table></div></details>')
    for index, (group, rows) in enumerate(view["groups"]):
        issues = sum(r["severity"] in ("warn", "bad") for r in rows)
        parts += [f'<details id="group-{index}" class="group" open><summary>{e(group)}<span>{len(rows)} settings' + (f' · {issues} to check' if issues else '') + '</span></summary>',
                  '<div class="table-scroll"><table><thead><tr><th>Setting</th><th>Configured</th><th>Runtime value</th><th>Source</th><th>Workers</th></tr></thead><tbody>']
        for row in rows:
            values = "".join(f'<li>Rank {e(rank)}: {e(value)}</li>' for rank, value in row["rank_values"])
            check = (f'<details class="rank-detail {row["agreement_severity"]}"><summary>{e(row["agreement"])}</summary><ul>{values}</ul></details>'
                     if values else f'<span class="agreement {row["agreement_severity"]}">{e(row["agreement"])}</span>')
            title = f' title="{e(row["origin"])}"' if row["origin"] else ''
            parts.append(f'<tr class="{row["severity"]}"><th scope="row">{e(row["label"])}</th><td>{e(row["configured"])}</td><td>{e(row["resolved"])}</td><td><span class="evidence"{title}>{e(row["evidence"])}</span></td><td>{check}<small class="comparison {row["comparison_severity"]}">{e(row["comparison"])}</small></td></tr>')
        parts.append('</tbody></table></div>')
        parts += [f'<p class="footnote">{e(note)}</p>' for note in group_notes(group, rows)]
        parts.append('</details>')
    parts.append('<details id="workers" class="group" open><summary>Workers<span>Draft tokens, KV transfer and kernel setup on each worker</span></summary><div class="table-scroll"><table><thead><tr><th>Rank</th><th>Draft tokens</th><th>KV transfer</th><th>Kernel setup</th></tr></thead><tbody>')
    parts += ['<tr>' + ''.join(f'<td>{e(value)}</td>' for value in row) + '</tr>' for row in rank_rows(view)]
    if not view["ranks"]:
        parts.append('<tr><td colspan="4">No worker reports</td></tr>')
    parts.append('</tbody></table></div></details>')
    parts.append('<details id="provenance" class="group"><summary>Build information<span>Image, packages and source commits</span></summary><dl>')
    parts += [f'<dt>{e(label)}</dt><dd>{e(value)}</dd>' for label, value in identity_items(view)]
    parts.append('</dl></details><footer>This page only reads settings and worker reports; it changes nothing. '
                 'A complete report does not mean the model\'s output quality or speed has been tested.</footer></main>')
    return view["title"], "\n".join(parts)
