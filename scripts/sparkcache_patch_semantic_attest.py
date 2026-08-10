#!/usr/bin/env python3
"""Structural attestation for SparkCache's two vLLM compatibility seams."""

from __future__ import annotations

import ast
import textwrap


class SemanticAttestationError(ValueError):
    pass


def attest_sources(scheduler_source: str, vmm_source: str) -> None:
    """Prove the required control flow from two inspected method sources."""

    def function(source: str, expected: str | tuple[str, ...]) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(source))
        found = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        allowed = (expected,) if isinstance(expected, str) else expected
        if len(found) != 1 or found[0].name not in allowed:
            raise SemanticAttestationError(
                f"expected one function named in {allowed!r}"
            )
        return found[0]

    def attr(node: ast.AST, base: str, name: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == name
            and isinstance(node.value, ast.Name)
            and node.value.id == base
        )

    def block_falls_through(statements: list[ast.stmt]) -> bool:
        falls_through = True
        for statement in statements:
            if not falls_through:
                return False
            falls_through = statement_falls_through(statement)
        return falls_through

    def statement_falls_through(statement: ast.stmt) -> bool:
        if isinstance(statement, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            return False
        if isinstance(statement, ast.If):
            if isinstance(statement.test, ast.Constant):
                selected = statement.body if statement.test.value else statement.orelse
                return block_falls_through(selected)
            return block_falls_through(statement.body) or block_falls_through(
                statement.orelse
            )
        if isinstance(statement, ast.While) and isinstance(
            statement.test, ast.Constant
        ):
            if not statement.test.value:
                return True
            # A constant-true loop cannot reach its successor unless a break
            # may target that loop. Nested-loop breaks do not count.
            def targets_loop(node: ast.AST) -> bool:
                if isinstance(node, ast.Break):
                    return True
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                        continue
                    if isinstance(child, ast.Break) or targets_loop(child):
                        return True
                return False

            return any(targets_loop(item) for item in statement.body)
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            return block_falls_through(statement.body)
        if isinstance(statement, (ast.Try, ast.TryStar)):
            if statement.finalbody and not block_falls_through(statement.finalbody):
                return False
            normal_path = block_falls_through(
                statement.body
            ) and block_falls_through(statement.orelse)
            handler_path = any(
                block_falls_through(handler.body) for handler in statement.handlers
            )
            return normal_path or handler_path
        if isinstance(statement, ast.Assert):
            return not (
                isinstance(statement.test, ast.Constant)
                and not statement.test.value
            )
        if isinstance(
            statement,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Delete,
                ast.Expr,
                ast.Pass,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.For,
                ast.AsyncFor,
            ),
        ):
            # These statements have a path to their successor.  In particular,
            # a for/async-for body may execute zero times.  Unsupported compound
            # control flow (for example Match) is deliberately not guessed.
            return True
        raise SemanticAttestationError(
            "cannot prove reachability across unsupported statement "
            f"{type(statement).__name__}"
        )

    def reachable_direct(statements: list[ast.stmt]):
        for index, statement in enumerate(statements):
            yield index, statement
            if not statement_falls_through(statement):
                break

    def nested_control_flow(statement: ast.stmt) -> list[ast.AST]:
        return [
            node
            for node in ast.walk(statement)
            if node is not statement
            and isinstance(
                node,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.TryStar,
                    ast.With,
                    ast.AsyncWith,
                    ast.Match,
                ),
            )
        ]

    def exact_no_failures_return(statement: ast.stmt) -> bool:
        return (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.UnaryOp)
            and isinstance(statement.test.op, ast.Not)
            and isinstance(statement.test.operand, ast.Name)
            and statement.test.operand.id == "total_failed_requests"
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and isinstance(statement.body[0].value, ast.Call)
            and isinstance(statement.body[0].value.func, ast.Name)
            and statement.body[0].value.func.id == "set"
            and not statement.body[0].value.args
            and not statement.body[0].value.keywords
        )

    def exact_failure_policy_return(statement: ast.stmt) -> bool:
        if not (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "should_fail"
            and not statement.orelse
        ):
            return False
        exits = [
            node
            for node in ast.walk(statement)
            if isinstance(node, (ast.Return, ast.Raise))
        ]
        return (
            len(exits) == 1
            and isinstance(exits[0], ast.Return)
            and isinstance(exits[0].value, ast.Name)
            and exits[0].value.id == "all_failed_req_ids"
            and statement.body[-1] is exits[0]
        )

    def prove_scheduler_prefix(statements: list[ast.stmt], loop_index: int) -> None:
        """Prove every path reaching recompute executes the repair loop.

        The only permitted exits before the inserted loop are the upstream
        no-failures fast return and the explicit failure-policy return. Any
        other exit or compound loop/try/match is unproved and fails closed.
        """
        approved_exits = 0
        for statement in statements[:loop_index]:
            if nested_control_flow(statement):
                raise SemanticAttestationError(
                    "unsupported compound control flow precedes scheduler repair"
                )
            exits = [
                node
                for node in ast.walk(statement)
                if isinstance(node, (ast.Return, ast.Raise))
            ]
            if not exits:
                continue
            if exact_no_failures_return(statement) or exact_failure_policy_return(
                statement
            ):
                approved_exits += 1
                continue
            raise SemanticAttestationError(
                "unapproved exit can bypass scheduler repair"
            )
        if approved_exits != 2 or not exact_failure_policy_return(
            statements[loop_index - 1]
        ):
            raise SemanticAttestationError(
                "scheduler repair must immediately follow the two exact upstream exits"
            )

    scheduler = function(scheduler_source, "_handle_invalid_blocks")
    loops = []
    for index, statement in reachable_direct(scheduler.body):
        if (
            not isinstance(statement, ast.For)
            or not isinstance(statement.target, ast.Name)
            or statement.target.id != "spark_req_id"
            or not isinstance(statement.iter, ast.BinOp)
        ):
            continue
        iterator = statement.iter
        if (
            isinstance(iterator.op, ast.BitOr)
            and isinstance(iterator.left, ast.Name)
            and iterator.left.id == "async_failed_req_ids"
            and isinstance(iterator.right, ast.Name)
            and iterator.right.id == "sync_failed_req_ids"
        ):
            loops.append((index, statement))
    if len(loops) != 1:
        raise SemanticAttestationError("missing exact async|sync invalid-block loop")
    loop_index, loop = loops[0]
    prove_scheduler_prefix(scheduler.body, loop_index)
    if loop.orelse:
        raise SemanticAttestationError("invalid-block repair loop must have no else")
    body = list(reachable_direct(loop.body))
    assignments = []
    spark_request_writes = []
    null_guards = []
    repairs = []
    for index, statement in body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "spark_request"
                for target in statement.targets
            )
        ):
            spark_request_writes.append(index)
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "spark_request"
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "get"
            and attr(statement.value.func.value, "self", "requests")
            and len(statement.value.args) == 1
            and isinstance(statement.value.args[0], ast.Name)
            and statement.value.args[0].id == "spark_req_id"
            and not statement.value.keywords
        ):
            assignments.append(index)
        if (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Name)
            and statement.test.left.id == "spark_request"
            and len(statement.test.ops) == len(statement.test.comparators) == 1
            and isinstance(statement.test.ops[0], ast.Is)
            and isinstance(statement.test.comparators[0], ast.Constant)
            and statement.test.comparators[0].value is None
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Continue)
            and not statement.orelse
        ):
            null_guards.append(index)
        if (
            isinstance(statement, ast.If)
            and attr(statement.test, "spark_request", "num_output_placeholders")
            and len(statement.body) == 2
            and not statement.orelse
        ):
            add, reset = statement.body
            if (
                isinstance(add, ast.AugAssign)
                and attr(add.target, "spark_request", "async_tokens_to_discard")
                and isinstance(add.op, ast.Add)
                and attr(add.value, "spark_request", "num_output_placeholders")
                and isinstance(reset, ast.Assign)
                and len(reset.targets) == 1
                and attr(reset.targets[0], "spark_request", "num_output_placeholders")
                and isinstance(reset.value, ast.Constant)
                and reset.value.value == 0
            ):
                repairs.append(index)
    if not (
        len(assignments)
        == len(spark_request_writes)
        == len(null_guards)
        == len(repairs)
        == 1
        and assignments[0] < null_guards[0] < repairs[0]
    ):
        raise SemanticAttestationError(
            "scheduler lookup, null guard, and exact repair must occur once in order"
        )
    returns = [
        index
        for index, statement in reachable_direct(scheduler.body)
        if isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Name)
        and statement.value.id == "sync_failed_req_ids"
    ]
    if not returns or loop_index >= returns[-1]:
        raise SemanticAttestationError(
            "discard/reset loop must precede final recompute return"
        )

    vmm = function(
        vmm_source,
        ("_validate_kv_transfer_vmm", "_verify_kv_transfer_compat"),
    )
    exact_guards = []
    reachable = list(reachable_direct(vmm.body))
    for index, statement in reachable:
        if not isinstance(statement, ast.If) or not isinstance(statement.test, ast.BoolOp):
            continue
        test = statement.test
        if not isinstance(test.op, ast.And) or len(test.values) != 2:
            continue
        first, second = test.values
        nonnull = (
            isinstance(first, ast.Compare)
            and len(first.ops) == len(first.comparators) == 1
            and isinstance(first.ops[0], ast.IsNot)
            and isinstance(first.comparators[0], ast.Constant)
            and first.comparators[0].value is None
            and isinstance(first.left, ast.Attribute)
            and first.left.attr == "kv_transfer_config"
            and isinstance(first.left.value, ast.Name)
            and first.left.value.id == "self"
        )
        connector = (
            isinstance(second, ast.Compare)
            and len(second.ops) == len(second.comparators) == 1
            and isinstance(second.ops[0], ast.Eq)
            and isinstance(second.comparators[0], ast.Constant)
            and second.comparators[0].value == "SparkContextCacheConnector"
            and isinstance(second.left, ast.Attribute)
            and second.left.attr == "kv_connector"
            and isinstance(second.left.value, ast.Attribute)
            and second.left.value.attr == "kv_transfer_config"
            and isinstance(second.left.value.value, ast.Name)
            and second.left.value.value.id == "self"
        )
        if (
            nonnull
            and connector
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and statement.body[0].value is None
            and not statement.orelse
        ):
            exact_guards.append(index)
    if len(exact_guards) != 1:
        raise SemanticAttestationError("missing exact SparkCache-only VMM return guard")
    guard_index = exact_guards[0]
    rejection_indices = [
        index
        for index, statement in reachable
        if any(isinstance(node, ast.Raise) for node in ast.walk(statement))
    ]
    if not rejection_indices or any(index <= guard_index for index in rejection_indices):
        raise SemanticAttestationError("SparkCache VMM exemption must dominate rejection")


def attest_kv_output_aggregator_source(source: str) -> None:
    """Prove worker load errors and completion IDs are aggregated together.

    SparkCache's provisional-startup safety depends on every worker's invalid
    block IDs reaching one scheduler output and on ``finished_recving`` being
    emitted only through the same all-worker aggregator. This attests the exact
    pinned runtime method instead of assuming generic vLLM behavior.
    """

    tree = ast.parse(textwrap.dedent(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "aggregate":
        raise SemanticAttestationError("expected exactly KVOutputAggregator.aggregate")
    function = functions[0]

    worker_loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model_runner_output"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "outputs"
    ]
    if len(worker_loops) != 1:
        raise SemanticAttestationError("missing exact all-worker output loop")
    loop = worker_loops[0]

    invalid_unions = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "invalid_block_ids"
        and isinstance(node.op, ast.BitOr)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "invalid_block_ids"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "kv_output"
    ]
    if len(invalid_unions) != 1:
        raise SemanticAttestationError("worker invalid block IDs are not unioned exactly")

    recv_updates = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "update_finished_set"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "finished_recving"
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "kv_output"
        and isinstance(node.args[1], ast.Attribute)
        and node.args[1].attr == "_recv_remaining_count"
        and isinstance(node.args[1].value, ast.Name)
        and node.args[1].value.id == "self"
        and isinstance(node.args[2], ast.Name)
        and node.args[2].id == "finished_recving"
    ]
    if len(recv_updates) != 1:
        raise SemanticAttestationError(
            "worker receive completion is not aggregated through the shared counter"
        )

    constructors = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "KVConnectorOutput"
    ]
    if len(constructors) != 1:
        raise SemanticAttestationError("expected one aggregated KVConnectorOutput")
    keywords = {item.arg: item.value for item in constructors[0].keywords if item.arg}
    invalid_value = keywords.get("invalid_block_ids")
    if not (
        isinstance(invalid_value, ast.Name)
        and invalid_value.id == "invalid_block_ids"
    ):
        raise SemanticAttestationError("aggregated invalid block IDs are not published")
    finished_value = keywords.get("finished_recving")
    if not (
        isinstance(finished_value, ast.BoolOp)
        and isinstance(finished_value.op, ast.Or)
        and len(finished_value.values) == 2
        and isinstance(finished_value.values[0], ast.Name)
        and finished_value.values[0].id == "finished_recving"
        and isinstance(finished_value.values[1], ast.Constant)
        and finished_value.values[1].value is None
    ):
        raise SemanticAttestationError(
            "aggregated receive completion is not published fail-closed"
        )
