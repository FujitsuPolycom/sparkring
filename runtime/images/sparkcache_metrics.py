"""Integrate connector log formatting without changing Prometheus observations."""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/metrics.py"

ORIGINAL = '''            xfer_metrics = self.transfer_stats_accumulator.reduce()
            xfer_metrics_str = ", ".join(f"{k}={v}" for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)
'''

REPLACEMENT = '''            formatter = getattr(
                self.transfer_stats_accumulator, "format_log_lines", None
            )
            if callable(formatter):
                for line in formatter():
                    log_fn("%s", line)
            else:
                xfer_metrics = self.transfer_stats_accumulator.reduce()
                xfer_metrics_str = ", ".join(
                    f"{k}={v}" for k, v in xfer_metrics.items()
                )
                log_fn("KV Transfer metrics: %s", xfer_metrics_str)
'''


def transform(raw):
    """Accept the pinned stock block or its already integrated equivalent."""
    source = raw.decode("utf-8")
    if source.count(REPLACEMENT) == 1 and ORIGINAL not in source:
        return raw
    if source.count(ORIGINAL) != 1 or REPLACEMENT in source:
        raise ValueError("Pinned KV metrics logger block differs")
    return source.replace(ORIGINAL, REPLACEMENT, 1).encode("utf-8")
