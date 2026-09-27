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
        ("Max sequences", "max_num_seqs", "max_num_seqs"),
        ("Quantization", None, "quantization"), ("Model dtype", None, "model_dtype"),
    ],
    "Prefill and attention": [
        ("Chunked prefill", "chunked_prefill_enabled", "chunked_prefill_enabled"),
        ("Batched token limit", "max_num_batched_tokens", "max_num_batched_tokens"),
        ("Activation quant fusion", None, "fuse_act_quant"),
        ("Attention backend", None, "attention_backend"),
        ("Resident decoder", None, "decoder_attention_implementations"),
        ("Q/K head dimensions", None, "qk_head_dims"),
        ("Value head dimensions", None, "value_head_dims"),
    ],
    "Decode and speculation": [
        ("Speculation method", None, "speculative_method"),
        ("Draft tokens", None, "num_speculative_tokens"),
        ("Draft model type", None, "draft_model_type"),
        ("Draft TP", None, "draft_tensor_parallel_size"),
        ("Draft loader", None, "draft_load_format"),
        ("Draft KV dtype", None, "draft_kv_cache_dtype"),
        ("Draft sampling", None, "draft_sample_method"),
        ("Rejection sampling", None, "rejection_sample_method"),
        ("Adaptive verification", None, "adaptive_verification"),
        ("CUDA graph mode", None, "cudagraph_mode"),
        ("Graph capture ceiling", None, "max_cudagraph_capture_size"),
        ("Linear backend", None, "linear_backend"), ("MoE backend", None, "moe_backend"),
        ("MXFP8 LM head", "VLLM_MXFP8_LM_HEAD", None),
        ("LM head A16", "VLLM_LM_HEAD_A16", None),
    ],
    "Transport": [
        ("RoCEnante switch", "VLLM_ENABLE_ROCE_ALLREDUCE", None),
        ("RoCE AR ceiling", "VLLM_ROCE_ALLREDUCE_MAX_SIZE", None),
        ("RoCE AG ceiling", "VLLM_ROCE_ALLGATHER_MAX_SIZE", None),
        ("NCCL PCI domains", "NCCL_IB_PRESERVE_PCI_DOMAIN", None),
        ("NCCL subnet routing", "NCCL_IB_SUBNET_AWARE_ROUTING", None),
        ("NCCL algorithm", "NCCL_ALGO", None),
        ("NCCL protocols", "NCCL_PROTO", None),
        ("NCCL channels min", "NCCL_MIN_NCHANNELS", None),
        ("NCCL channels max", "NCCL_MAX_NCHANNELS", None),
    ],
    "Cache and loader": [
        ("KV transfer", None, "kv_transfer_enabled"),
        ("KV connector", None, "kv_connector"),
        ("GPU prefix cache", "prefix_caching_enabled", "prefix_caching_enabled"),
        ("KV dtype", None, "cache_dtype"),
        ("KV memory per rank", None, "kv_cache_memory_bytes"),
        ("Block size", "block_size", "block_size"),
        ("Mamba cache mode", None, "mamba_cache_mode"),
        ("Checkpoint policy", None, "recurrent_checkpoint_policy"),
        ("Loader", "load_format", "load_format"),
        ("Read mode", None, "loader_read_mode"), ("I/O threads", None, "loader_io_threads"),
    ],
}
QWEN = [
    ("HC prefill ownership", "VLLM_QWEN3_8_HC_PREFILL_MODE", "hc_prefill_row_ownership"),
    ("HC projection ranks", None, "hc_projection_tp_size"),
    ("Qwen coalescing", "VLLM_QWEN3_8_PREFILL_COALESCE", None),
    ("Compact MTP", "VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT", None),
    ("Projection overlap", "VLLM_QWEN3_8_FLASH_NEXT_OVERLAP", None),
    ("MTP NVFP4 head", "VLLM_MTP_NVFP4_LM_HEAD", None),
    ("MoE input scaling", "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE", None),
    ("GDN metadata fastpath", "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH", None),
]
MIMO = [
    ("FP4 MoE A16", "VLLM_B12X_MOE_FP4_FORCE_A16", None),
    ("Aux MXFP8 streaming", "VLLM_DFLASH_AUX_MXFP8_STREAMING", None),
    ("Aux BF16 staging", "VLLM_DFLASH_AUX_BF16_STAGING", None),
    ("Compact RoPE", "VLLM_DFLASH_COMPACT_ROPE", None),
    ("Sharded aux projection", "VLLM_DFLASH_SHARD_AUX_PROJECTION", None),
]
NOTE = ("Agreement means workers report the same setting, not that a kernel executed. "
        "Configured and prepared values are not request execution evidence. Unknown is not OFF.")
ENV_RESOLVED = {
    'VLLM_ENABLE_ROCE_ALLREDUCE': 'tp_rocenante_enabled',
    'VLLM_ROCE_ALLREDUCE_MAX_SIZE': 'tp_roce_allreduce_max_bytes',
    'VLLM_ROCE_ALLGATHER_MAX_SIZE': 'tp_roce_allgather_max_bytes',
    'VLLM_MXFP8_LM_HEAD': 'target_head_mxfp8',
    'VLLM_LM_HEAD_A16': 'target_head_a16',
    'VLLM_MTP_NVFP4_LM_HEAD': 'draft_head_nvfp4',
}
LOCAL_FIELDS = {'tp_roce_hcas', 'tp_roce_gid_index', 'tp_roce_pci_domains', 'tp_nccl_library_path'}
AUTO_FIELDS = {'model_dtype', 'quantization', 'attention_backend', 'draft_tensor_parallel_size',
               'draft_kv_cache_dtype', 'draft_sample_method', 'rejection_sample_method',
               'cudagraph_mode', 'gdn_prefill_backend', 'max_cudagraph_capture_size',
               'fuse_act_quant', 'max_model_len', 'max_num_seqs', 'max_num_batched_tokens',
               'block_size', 'kv_cache_memory_bytes', 'prefix_caching_enabled', 'chunked_prefill_enabled',
               'draft_load_format', 'linear_backend', 'moe_backend'}
RUNTIME_ONLY = {'served_model_name': 'From launch arguments',
                'model_type': 'From checkpoint', 'decoder_attention_modules': 'Runtime sample',
                'sampled_qk_head_dims': 'From model', 'sampled_value_head_dims': 'From model',
                'hc_projection_tp_size': 'From topology', 'target_head_quantization': 'Runtime selection',
                'draft_head_quantization': 'Runtime selection', 'draft_head_shared': 'Runtime selection'}
GROUPS['Prefill and attention'] = [
    (label, cfg, {'decoder_attention_implementations': 'decoder_attention_modules',
                  'qk_head_dims': 'sampled_qk_head_dims', 'value_head_dims': 'sampled_value_head_dims'}.get(actual, actual))
    for label, cfg, actual in GROUPS['Prefill and attention']]
GROUPS['Prefill and attention'].append(('GDN prefill backend', 'gdn_prefill_backend', 'gdn_prefill_backend'))
GROUPS['Transport'] = [
    ('RoCEnante available (TP)', 'VLLM_ENABLE_ROCE_ALLREDUCE', 'tp_rocenante_enabled'),
    ('RoCE AR ceiling', 'VLLM_ROCE_ALLREDUCE_MAX_SIZE', 'tp_roce_allreduce_max_bytes'),
    ('RoCE AG shard ceiling', 'VLLM_ROCE_ALLGATHER_MAX_SIZE', 'tp_roce_allgather_max_bytes'),
    ('NCCL runtime version (TP)', None, 'tp_nccl_version'),
    ('NCCL library handle (TP)', None, 'tp_nccl_library_path'),
    ('RoCEnante selected HCAs', 'B12X_ROCE_HCA', 'tp_roce_hcas'),
    ('RoCEnante GID index', None, 'tp_roce_gid_index'),
    ('RoCEnante PCI domains', None, 'tp_roce_pci_domains'),
    ('PCI-domain routing preference', 'NCCL_IB_PRESERVE_PCI_DOMAIN', None),
    ('NCCL subnet-routing override', 'NCCL_IB_SUBNET_AWARE_ROUTING', None),
    ('NCCL algorithm override', 'NCCL_ALGO', None),
    ('NCCL protocol override', 'NCCL_PROTO', None),
    ('NCCL channel lower bound', 'NCCL_MIN_NCHANNELS', None),
    ('NCCL channel upper bound', 'NCCL_MAX_NCHANNELS', None),
]
GROUPS['Decode and speculation'] += [
    ('Target head precision', None, 'target_head_quantization'),
    ('Draft head precision', None, 'draft_head_quantization'),
    ('Draft shares target head', None, 'draft_head_shared'),
]


def clean(value):
    """Prevent terminal control sequences in values and keep one fact per line."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", "?", str(value))


def known(record):
    return isinstance(record, dict) and record.get("state") == "known" and "value" in record


def fact_value(record):
    return record["value"] if known(record) else None


def display(record, key=""):
    if not known(record):
        return {'not_set_in_environment': 'Not set in environment', 'invalid_value': 'Invalid value',
                'not_supplied': 'Not supplied', 'per_collective_or_not_exposed': 'Not exposed by NCCL',
                'not_collected': 'Not collected'}.get(record.get('reason'), 'Unknown')
    value = record["value"]
    if value is None:
        if record.get('reason') == 'not_applicable':
            return 'Not applicable'
        return 'Auto / inherited' if key in AUTO_FIELDS else 'None'
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


def signature(record):
    value = record.get('value')
    if type(value) is str:
        value = value.removeprefix('torch.')
    return json.dumps(value, sort_keys=True)


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
        rows = []
        for label, config_key, actual_key in fields:
            actual_key = actual_key or ENV_RESOLVED.get(config_key)
            config_key = config_key or (actual_key if actual_key in ARG_FIELDS else None)
            requested = configured(doc, config_key)
            actual = effective(doc, actual_key)
            reports = [(r.get("identity", {}).get("rank", {}),
                        effective(r, actual_key) if actual_key else configured(r, config_key))
                       for r in ranks]
            available = [value for _, value in reports if known(value)]
            distinct = {signature(value) for value in available}
            severity = "neutral"
            if actual_key in LOCAL_FIELDS and available:
                agreement = f'Per node · {len(available)}/{expected or "?"} reported'
            elif len(distinct) > 1:
                agreement, severity = "Workers differ", "bad"
            elif available and expected and len(available) == expected and unique_ranks:
                agreement = f"{len(available)}/{expected} agree"
            elif available:
                agreement, severity = f"{len(available)}/{expected or '?'} reported", "warn"
            else:
                unset = sum(value.get('reason') == 'not_set_in_environment' for _, value in reports)
                agreement = f'Not set on {unset}/{expected or "?"}' if unset else f"Unknown on all {len(ranks)} ranks" if ranks else "No worker reports"
            if not known(actual) and len(distinct) == 1:
                actual = available[0] if actual_key else actual
            comparison = []
            automatic = config_key in AUTO_FIELDS and known(requested) and requested['value'] in (None, 'auto')
            if actual_key not in LOCAL_FIELDS and not automatic and known(requested) and known(actual) and signature(requested) != signature(actual):
                comparison.append("Adjusted for alignment" if actual_key == "block_size" else "Configured != resolved")
                severity = "warn" if actual_key == "block_size" else "bad"
            reference = actual if actual_key else requested
            if actual_key not in LOCAL_FIELDS and known(reference) and any(signature(reference) != signature(value) for value in available):
                comparison.append("API != workers")
                severity = "bad"
            source = actual.get('source', '')
            evidence = ('Resident runtime' if source.startswith('resident_') else 'Resolved configuration') if known(actual) else "Configured only" if known(requested) or available else "Not reported"
            resolved_label = display(actual, actual_key or '') if actual_key else 'No runtime confirmation'
            if actual_key in LOCAL_FIELDS and len(distinct) > 1:
                resolved_label = 'See per-node values'
            rows.append({"label": label, "configured": display(requested, config_key or "") if config_key else RUNTIME_ONLY.get(actual_key, 'Runtime derived'),
                         "resolved": resolved_label,
                         "evidence": evidence, "agreement": agreement, "severity": severity,
                         "comparison": "; ".join(comparison),
                         'source': source, 'reason': actual.get('reason') or requested.get('reason'),
                         "rank_values": [(display(rank), display(value, actual_key or config_key or "")) for rank, value in reports]})
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
            "state": state, "stale": stale, "complete": complete,
            "age": f"{age:.2f}s" if isinstance(age, (int, float)) else "Not collected",
            "groups": groups, "ranks": ranks,
            "issue_count": sum(row["severity"] in ("warn", "bad") for _, rows in groups for row in rows)}


def render_text(doc):
    view = summarize(doc)
    lines = [view["title"], f"Model: {view['model']} | {view['topology']}",
             f"Model architecture: {view['architecture']}",
             f"Workers: {view['workers']} | reporting: {view['state']} | stale: {view['stale']} | snapshot age: {view['age']}", "", NOTE]
    for section in detail_sections(doc, view):
        lines += ["", section['title'].upper(), section['note'], " | ".join(section['headers'])]
        lines += [" | ".join(clean(value) for value in row) for row in section['rows']] or ['Not reported']
    for group, rows in view["groups"]:
        lines += ["", group.upper()]
        table = [("SETTING", "CONFIGURED", "RESOLVED", "EVIDENCE", "RANK CHECK")]
        table += [(r["label"], r["configured"], r["resolved"], r["evidence"],
                   r["agreement"] + ("; " + r["comparison"] if r["comparison"] else "")) for r in rows]
        widths = [max(len(row[i]) for row in table) for i in range(4)]
        lines += ["  ".join(value.ljust(widths[i]) if i < 4 else value for i, value in enumerate(row)) for row in table]
        for row in rows:
            if row["severity"] == "bad":
                lines.append("  " + row["label"] + ": " + "; ".join(f"rank {rank}={value}" for rank, value in row["rank_values"]))
    lines += ["", "WORKER PREPARATION", "Rank | Draft tokens | KV transfer | Preparation (not execution)"]
    lines += [" | ".join(row) for row in rank_rows(view)] or ["No worker reports"]
    lines += ["", "SOURCE IDENTITY"]
    lines += [f"{key}: {clean(value)}" for key, value in identity_items(view)]
    return ("\n".join(lines) + "\n").replace("—", "--").replace(" · ", " | ")


def rank_rows(view):
    result = []
    for rank in view["ranks"]:
        preparation = effective(rank, "kernel_preparation")
        state = preparation.get("session_state", preparation.get("state", "unknown"))
        if rank.get("error"):
            state = "Snapshot unavailable"
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
            items.append((component + " source", source["commit"]))
    if provenance.get("composition_sha256"):
        items.append(("Composition", provenance["composition_sha256"]))
    return items or [("Provenance", "Not reported")]


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
    reasons = {'needs_previous_snapshot': 'Waiting for a second snapshot',
               'no_drafts_in_window': 'No drafts in this window', 'counter_reset': 'Counters restarted',
               'counters_unavailable': 'Counters unavailable',
               'inconsistent_counter_snapshot': 'Counter snapshot inconsistent'}
    for key, label in (('recent', 'Since previous refresh'), ('lifetime', 'Since engine start')):
        values = acceptance.get(key, {})
        if values.get('state') == 'known':
            rate = values.get('acceptance_rate')
            acceptance_rows.append((label, numeric(values.get('rounds'), digits=0),
                numeric(rate * 100, '%') if type(rate) in (int, float) else 'Unknown',
                numeric(values.get('estimated_tokens_per_verification'), digits=2),
                numeric(values.get('window_seconds'), ' s') if key == 'recent' else 'Engine lifetime'))
        else:
            reason = values.get('reason', acceptance.get('reason'))
            acceptance_rows.append((label, reasons.get(reason, 'Not observed'), 'Not observed',
                                    'Not observed', 'Not observed'))
    section('acceptance', 'Draft acceptance',
        'Prometheus engine counters. Tokens per verification is estimated as 1 + accepted drafts per round; '
        'it is not tok/s. Acceptance varies with prompts and sampling.',
        ('Window', 'Verification rounds', 'Draft acceptance', 'Tokens / verification', 'Duration'), acceptance_rows)
    positions = acceptance.get('lifetime', {}).get('positions', [])
    if positions:
        section('acceptance-positions', 'Acceptance by draft position', 'Cumulative engine counters since startup.',
            ('Draft position', 'Drafted', 'Accepted', 'Acceptance'),
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
            scope = ('Remote / unrecognized filesystem; not probed' if reason == 'remote_or_unrecognized_filesystem_not_probed_in_worker'
                     else 'Mounted filesystem' if disk.get('state') == 'known' else 'Not available')
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
                                 library.get('version') or 'Not encoded in filename', library.get('path', 'Unknown')))
        transport = rank.get('transport') or {}
        for name, group in transport.get('groups', {}).items():
            if group.get('state') != 'known':
                continue
            roce, nccl = group.get('rocenante', {}), group.get('nccl', {})
            if (fact_value(group.get('world_size')) == 1
                    and fact_value(roce.get('enabled')) is not True
                    and fact_value(nccl.get('available')) is not True):
                singleton_rows.append((label, name, '1', 'Not applicable (one rank)'))
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

    section('nodes', 'Node memory',
        'Host used = total − available. On Spark, CPU and GPU share this host memory pool. '
        'Cgroup memory is shown separately and must not be added to host usage.',
        ('Rank', 'Worker hostname', 'Host used', 'Host available', 'Host total', 'Swap used', 'Cgroup current / limit', 'Sampled'), memory_rows)
    section('storage', 'Filesystem space',
        'Available space is usable by an unprivileged process. Repeated local filesystems are grouped. '
        'Remote mounts are not probed from inference workers.',
        ('Rank', 'Mount paths', 'Available', 'Used', 'Total', 'Scope'), disk_rows)
    section('versions', 'CUDA and NCCL versions',
        'Torch build CUDA, installed toolkit and loaded runtime are separate. Adding a newer toolkit '
        'does not rebuild Torch or its native extensions.',
        ('Rank', 'Host NVIDIA driver', 'Torch build CUDA', 'CUDA toolkit', 'NVCC', 'Resident TP NCCL'), version_rows)
    section('libraries', 'Mapped runtime libraries',
        'Files resident in each worker, read from /proc/self/maps. Version text comes from filenames; '
        'the resident communicator reports the NCCL runtime version above.',
        ('Rank', 'Component', 'Filename version', 'Mapped path'), library_rows, False)
    section('packages', 'Installed packages', 'Distribution metadata; this is separate from source commits and runtime libraries.',
        ('Rank', 'Package', 'Version'), package_rows, False)
    section('communicators', 'Resident transport groups',
        'Availability and size limits describe constructed communicators. They do not establish which '
        'backend a request used. NCCL algorithm, protocol and channels are not exposed by these objects.',
        ('Rank', 'Group', 'Size', 'RoCEnante available', 'NCCL available', 'AR ceiling', 'AG shard ceiling', 'Selected HCAs'), group_rows)
    if singleton_rows:
        section('singleton-groups', 'Single-rank groups',
            'These groups contain one rank and report no available cross-rank communicator. '
            'No inter-rank transport is needed; this is not a network failure.',
            ('Rank', 'Group', 'Size', 'Inter-rank transport'), singleton_rows, False)
    section('links', 'Selected NICs and link rates',
        'Negotiated link rate is not measured payload bandwidth. PCI domain is the first part of the PCI address. '
        'NIC functions with the same PCI device key share a physical uplink; do not sum their capacities.',
        ('Rank', 'HCA', 'PCI address', 'Interface', 'State', 'Link rate', 'Current PCIe link', 'MTU'), nic_rows)
    section('addresses', 'NIC identifiers',
        'Only selected resident HCAs are inspected. IP addresses are not exposed by this passive worker collector.',
        ('Rank', 'Interface', 'MAC', 'IP addresses', 'PCI device key'), address_rows, False)
    section('rdma-counters', 'RDMA port traffic',
        'Host cumulative port counters, not process traffic. Data counters are converted from 4-byte words '
        'to bytes. No bandwidth or latency test is run by this dashboard.',
        ('Rank', 'HCA', 'Port', 'State', 'Transmitted', 'Received', 'Receive errors', 'Transmit discards'), counter_rows, False)
    return sections


def render_report(doc):
    """Escape every runtime value; only static markup enters the HTML response."""
    view = summarize(doc)
    e = lambda value: escape(clean(value), quote=True)
    state_class = "good" if view["complete"] else "warn"
    parts = [f'<main id="report"><header><p class="eyebrow">CLUSTER / RUNTIME STATUS</p><h1>{e(view["title"])}</h1>',
             f'<p class="subtitle">{e(view["model"])} <span>·</span> {e(view["topology"])}</p>',
             f'<p class="subtitle">Model architecture: {e(view["architecture"])}</p></header>',
             '<div class="cards">',
             f'<section class="card"><span>Worker reports</span><strong>{e(view["workers"])}</strong><small class="{state_class}">{e(view["state"])}{ " · stale" if view["stale"] else ""}</small></section>',
             f'<section class="card"><span>Snapshot age</span><strong>{e(view["age"])}</strong><small>Cached worker metadata</small></section>',
             f'<section class="card"><span>Settings to inspect</span><strong>{view["issue_count"]}</strong><small>Differences, adjustments or partial reports</small></section></div>',
             f'<p class="notice">{e(NOTE)}</p>']
    parts.append('<nav class="section-links" aria-label="Report sections"><a href="#group-0">Settings</a>'
                 '<a href="#nodes">Memory and disk</a><a href="#communicators">Transport</a>'
                 '<a href="#versions">Versions</a><a href="#acceptance">Acceptance</a></nav>')
    parts.append('<p class="legend">Configured includes parsed launch defaults. Runtime derived means no separate launch setting. '
                 'Not set in environment does not rule out configuration files or internal defaults. '
                 'Unknown and not collected mean evidence is missing.</p>')
    for section in detail_sections(doc, view):
        opened = ' open' if section['open'] else ''
        parts += [f'<details id="{e(section["id"])}" class="group"{opened}><summary>{e(section["title"])}</summary>',
                  f'<p class="section-note">{e(section["note"])}</p>', '<div class="table-scroll"><table><thead><tr>',
                  ''.join(f'<th scope="col">{e(label)}</th>' for label in section['headers']), '</tr></thead><tbody>']
        for row in section['rows']:
            parts.append('<tr>' + ''.join(f'<td>{e(value)}</td>' for value in row) + '</tr>')
        if not section['rows']:
            parts.append(f'<tr><td colspan="{len(section["headers"])}">Not reported by this snapshot</td></tr>')
        parts.append('</tbody></table></div></details>')
    for index, (group, rows) in enumerate(view["groups"]):
        issues = sum(r["severity"] in ("warn", "bad") for r in rows)
        parts += [f'<details id="group-{index}" class="group" open><summary>{e(group)}<span>{len(rows)} settings' + (f' · {issues} to inspect' if issues else '') + '</span></summary>',
                  '<div class="table-scroll"><table><thead><tr><th>Setting</th><th>Configured</th><th>Resolved</th><th>Evidence</th><th>Ranks</th></tr></thead><tbody>']
        for row in rows:
            values = "".join(f'<li>Rank {e(rank)}: {e(value)}</li>' for rank, value in row["rank_values"])
            check = f'<details class="rank-detail"><summary>{e(row["agreement"])}</summary><ul>{values}</ul></details>' if values else e(row["agreement"])
            source = row['source'] + (' · ' + row['reason'] if row['reason'] else '')
            parts.append(f'<tr class="{row["severity"]}"><th scope="row">{e(row["label"])}</th><td>{e(row["configured"])}</td><td>{e(row["resolved"])}</td><td><span class="evidence" title="{e(source)}">{e(row["evidence"])}</span></td><td>{check}<small class="comparison">{e(row["comparison"])}</small></td></tr>')
        parts.append('</tbody></table></div></details>')
    parts.append('<details id="workers" class="group" open><summary>Worker preparation<span>Resident state, not execution evidence</span></summary><div class="table-scroll"><table><thead><tr><th>Rank</th><th>Draft tokens</th><th>KV transfer</th><th>Preparation</th></tr></thead><tbody>')
    parts += ['<tr>' + ''.join(f'<td>{e(value)}</td>' for value in row) + '</tr>' for row in rank_rows(view)]
    if not view["ranks"]:
        parts.append('<tr><td colspan="4">No worker reports</td></tr>')
    parts.append('</tbody></table></div></details>')
    parts.append('<details id="provenance" class="group"><summary>Source identity<span>Image, packages and source commits</span></summary><dl>')
    parts += [f'<dt>{e(label)}</dt><dd>{e(value)}</dd>' for label, value in identity_items(view)]
    parts.append('</dl></details><footer>Read-only configuration and passive worker reports. Reporting completeness is not an inference correctness or performance test.</footer></main>')
    return view["title"], "\n".join(parts)
