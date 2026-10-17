"""Model-neutral mechanics for reproducible SparkCache deployments."""

from .command import (
    compact_json,
    drop_option,
    integer_option,
    one_option,
    option_values,
    optional_one_option,
    vllm_arguments,
)
from .container import (
    build_container_command,
    launch_container,
    normalized_posix_path,
)
from .errors import DeploymentContractError
from .inspection import environment_map, read_single_inspection
from .ports import validate_port
from .patches import apply_verified_patch
from .receipts import validate_overlay_receipt
from .source import file_sha256, source_tree_sha256
from .semantic import (
    AssistantCompletion,
    SemanticGateInconclusive,
    assistant_completion,
    assistant_content,
    build_long_prompt,
    request_chat,
    run_semantic_hit,
    run_semantic_miss,
)

__all__ = (
    "AssistantCompletion",
    "DeploymentContractError",
    "SemanticGateInconclusive",
    "apply_verified_patch",
    "assistant_completion",
    "assistant_content",
    "build_long_prompt",
    "build_container_command",
    "compact_json",
    "drop_option",
    "environment_map",
    "file_sha256",
    "integer_option",
    "launch_container",
    "normalized_posix_path",
    "one_option",
    "option_values",
    "optional_one_option",
    "read_single_inspection",
    "request_chat",
    "run_semantic_hit",
    "run_semantic_miss",
    "source_tree_sha256",
    "validate_port",
    "validate_overlay_receipt",
    "vllm_arguments",
)
