"""CPU reduction and model-boundary tests of packaged methods; no GPU qualification."""

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
from pathlib import Path
import sys
from threading import Barrier
from types import SimpleNamespace as NS
import unittest
from types import ModuleType
from unittest.mock import patch
from typing import cast

import torch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "mhc_regression_installer", HERE / "install.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
_, SOURCES = installer.package(preimages=True)
_, CANDIDATE = installer.package()
helper = ModuleType("mhc_prefill_regression_helper")
sys.modules[helper.__name__] = helper
exec(
    compile(
        CANDIDATE["vllm/models/glm5next/nvidia/mhc_prefill_sharding.py"],
        "mhc_prefill_sharding.py",
        "exec",
    ),
    helper.__dict__,
)


def load_method(path, cls, name, extras=None):
    tree = ast.parse(path)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = copy.deepcopy(
        next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    f.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, f], type_ignores=[]))
    scope = {"torch": torch, "cast": cast}
    scope.update(extras or {})
    exec(compile(module, "<packaged runtime method>", "exec"), scope)
    return scope[name]


class Fabric:
    def __init__(self):
        self.barrier = Barrier(4, timeout=15)
        self.values = [None] * 4
        self.calls = [[] for _ in range(4)]

    def exchange(self, rank, kind, x):
        self.calls[rank].append((kind, tuple(x.shape)))
        self.values[rank] = x.clone()
        self.barrier.wait()
        if kind == "ag":
            result = torch.cat(self.values, dim=0)
        else:
            result = torch.stack(self.values).sum(dim=0)
            if kind == "rs":
                result = result.chunk(4, dim=0)[rank].clone()
        self.barrier.wait()
        return result


class Owner:
    def __init__(self, fabric, rank):
        self.fabric, self.rank = fabric, rank
        self.rs_count = self.ag_count = 0

    def local_view(self, x):
        return x.chunk(4, dim=0)[self.rank]

    def reduce_scatter(self, x):
        self.rs_count += 1
        return self.fabric.exchange(self.rank, "rs", x)

    def all_gather(self, x):
        self.ag_count += 1
        return self.fabric.exchange(self.rank, "ag", x)

    def finish(self, layers, auxiliary_gathers):
        assert self.rs_count == layers * 2
        assert self.ag_count == layers * 2 + auxiliary_gathers


def pre(x, *args, **kwargs):
    n, h = x.shape
    residual = x[:, None].expand(n, 4, h).clone()
    return residual, torch.ones(n, 4), torch.ones(n, 4, 4), x * 2


def post_pre(x, residual, post, comb, *args, **kwargs):
    updated = residual * 0.5 + x[:, None]
    return updated, post + 0.25, comb + 0.5, updated.sum(1) * 0.25


def post(x, residual, post, comb):
    return residual + x[:, None]


class BoundaryTest(unittest.TestCase):
    def run_model(self, candidate, owned):
        rel = "models/glm5next/nvidia/model.py"
        path = (CANDIDATE if candidate else SOURCES)["vllm/" + rel]
        fabric = Fabric()
        pp = NS(is_first_rank=True, is_last_rank=True)
        globals_ = {
            "get_pp_group": lambda: pp,
            "hc_contract": lambda x, n: x.sum(1) * 0.25,
            "maybe_create_mhc_prefill_ownership": lambda m, x, p: m.owner,
        }
        decoder = load_method(path, "Glm5NextDecoderLayer", "forward", globals_)
        forward = load_method(path, "Glm5NextModel", "forward", globals_)
        auxiliary = load_method(
            path, "Glm5NextModel", "_prepare_aux_hidden_state", globals_
        )

        class Layer:
            __call__ = decoder

        def run(rank):
            owner = Owner(fabric, rank) if owned else None
            layers = []
            for index in range(3):
                layer = Layer()
                layer.mhc = True
                layer.is_mtp_layer = False
                layer.is_sequence_parallel = False
                layer.layer_idx = index
                layer.num_hidden_layers = 3
                layer.n = 4
                layer._mlp_is_moe = index > 0
                layer._b12x_mhc = NS(run_pre=pre)
                layer.hc_attn_fn_broadcast = torch.ones(1)
                for kind in ["attn", "ffn"]:
                    setattr(layer, f"hc_{kind}_fn", torch.ones(1))
                    setattr(layer, f"hc_{kind}_scale", torch.ones(1))
                    setattr(layer, f"hc_{kind}_base", torch.ones(1))
                layer.input_layernorm = NS(weight=torch.ones(4), variance_epsilon=1e-5)
                layer.post_attention_layernorm = layer.input_layernorm
                layer.hc_fused_post_pre = post_pre
                layer.hc_post = post

                def attention(hidden_states, positions, *, defer_tp_reduction=False):
                    assert hidden_states.shape[0] == positions.shape[0] == 8
                    partial = hidden_states * ((rank + 1) / 16)
                    return (
                        partial
                        if defer_tp_reduction
                        else fabric.exchange(rank, "ar", partial)
                    )

                def mlp(
                    x, already_sequence_parallel=False, *, defer_tp_reduction=False
                ):
                    assert x.shape == (8, 4) and not already_sequence_parallel
                    partial = x * ((rank + 1) / 32)
                    return (
                        partial
                        if defer_tp_reduction
                        else fabric.exchange(rank, "ar", partial)
                    )

                layer.self_attn = attention
                layer.mlp = mlp
                layers.append(layer)
            model = NS(
                owner=owner,
                is_sequence_parallel=False,
                start_layer=0,
                _active_layers=layers,
                aux_hidden_state_layers=[0, 1, 3],
                dflash_capture=False,
                norm=lambda x: x * 0.5,
            )
            model._prepare_aux_hidden_state = lambda *args: auxiliary(model, *args)
            x = torch.arange(32, dtype=torch.float64).reshape(8, 4) / 32
            result = forward(model, None, torch.arange(8), None, inputs_embeds=x)
            return result, owner

        with ThreadPoolExecutor(max_workers=4) as pool:
            result = list(pool.map(run, range(4)))
        return result, fabric.calls

    def test_actual_model_methods_keep_full_boundaries_and_local_residual(self):
        baseline, baseline_calls = self.run_model(False, False)
        default, default_calls = self.run_model(True, False)
        sharded, calls = self.run_model(True, True)
        self.assertEqual(default_calls, baseline_calls)
        for rank in range(4):
            for actual in [default[rank][0], sharded[rank][0]]:
                self.assertTrue(torch.equal(actual[0], baseline[rank][0][0]))
                self.assertEqual(actual[0].shape, (8, 4))
                for got, want in zip(actual[1], baseline[rank][0][1]):
                    self.assertTrue(torch.equal(got, want))
                    self.assertEqual(got.shape[0], 8)
            self.assertEqual(sharded[rank][1].rs_count, 6)
            self.assertEqual(sharded[rank][1].ag_count, 8)  # six steady-state + two aux
            self.assertNotIn("ar", [kind for kind, shape in calls[rank]])
            self.assertEqual(calls[rank][0][0], "rs")  # no initial boundary AG

    def test_mtp_rejects_owner_before_compute(self):
        path = CANDIDATE["vllm/models/glm5next/nvidia/model.py"]
        fn = load_method(path, "Glm5NextDecoderLayer", "forward")
        with self.assertRaisesRegex(RuntimeError, "MTP"):
            fn(
                NS(mhc=True, is_mtp_layer=True),
                None,
                None,
                mhc_prefill_ownership=object(),
            )


class DeferralTest(unittest.TestCase):
    def test_actual_row_parallel_reduces_once_or_returns_partial_and_keeps_tuple(self):
        path = CANDIDATE["vllm/model_executor/layers/linear.py"]
        calls = []
        fn = load_method(
            path,
            "RowParallelLinear",
            "forward",
            {"tensor_model_parallel_all_reduce": lambda x: calls.append(x) or x * 4},
        )
        obj = NS(
            input_is_parallel=True,
            tp_rank=0,
            tp_size=4,
            skip_bias_add=False,
            bias=None,
            quant_method=NS(apply=lambda self, x, bias: x * 2),
            reduce_results=True,
            return_bias=True,
        )
        x = torch.arange(8).reshape(2, 4)
        self.assertTrue(torch.equal(fn(obj, x)[0], x * 8))
        self.assertEqual(len(calls), 1)
        result, bias = fn(obj, x, defer_tp_reduction=True)
        self.assertTrue(torch.equal(result, x * 2))
        self.assertIsNone(bias)
        self.assertEqual(len(calls), 1)
        self.assertTrue(obj.reduce_results)
        obj.reduce_results = False
        with self.assertRaises(RuntimeError):
            fn(obj, x, defer_tp_reduction=True)

    def runner(self, *, reduced=False, transform=None):
        path = CANDIDATE["vllm/model_executor/layers/fused_moe/runner/moe_runner.py"]
        calls = []
        globals_ = {
            "tensor_model_parallel_all_reduce": lambda x: calls.append("ar") or x * 4,
            "_unpack": lambda x: x,
        }
        obj = NS(
            moe_config=NS(
                tp_size=4,
                dp_size=1,
                ep_size=1,
                pcp_size=1,
                is_sequence_parallel=False,
                skip_final_all_reduce=False,
                hidden_dim_unpadded=0,
                moe_parallel_config=NS(use_all2all_kernels=False),
            ),
            _fused_output_is_reduced=reduced,
            routed_input_transform=None,
            routed_output_transform=transform,
            router=NS(),
            _quant_method=NS(has_unpadded_output=False),
        )
        obj.apply_routed_input_transform = lambda x: (x, x)
        obj._maybe_pad_hidden_states = lambda shared, x: (x, None, 3)
        obj._forward_entry = lambda *args: (
            calls.append("compute")
            or (torch.full((2, 4), 2.0), torch.full((2, 4), 3.0))
        )
        obj._encode_layer_name = lambda: "test"
        obj._maybe_apply_routed_scale_to_output = lambda s, f: (s, f)
        obj.apply_routed_output_transform = lambda x: x
        obj._maybe_add_zero_expert_output = lambda x: x
        for name in [
            "_maybe_reduce_routed_output_before_transform",
            "_maybe_reduce_shared_expert_output",
            "_maybe_reduce_final_output",
        ]:
            f = load_method(path, "MoERunner", name, globals_)
            setattr(obj, name, lambda *args, _f=f, **kwargs: _f(obj, *args, **kwargs))
        return obj, load_method(path, "MoERunner", "forward", globals_), calls

    def test_actual_runner_merges_shared_routed_before_deferred_reduction_and_truncates(
        self,
    ):
        module = "vllm.models.glm5next.nvidia.mhc_prefill_sharding"
        for defer, expected, calls_expected in [
            (False, 20.0, ["compute", "ar"]),
            (True, 5.0, ["compute"]),
        ]:
            obj, fn, calls = self.runner()
            with patch.dict(sys.modules, {module: helper}):
                result = fn(
                    obj, torch.ones(2, 4), torch.ones(2, 4), defer_tp_reduction=defer
                )
            self.assertEqual(result.shape, (2, 3))
            self.assertTrue(torch.all(result == expected))
            self.assertEqual(calls, calls_expected)
            self.assertFalse(obj.moe_config.skip_final_all_reduce)

    def test_unsupported_moe_rejected_before_any_compute_or_collective(self):
        module = "vllm.models.glm5next.nvidia.mhc_prefill_sharding"
        for reduced, transform in [(True, None), (False, object())]:
            obj, fn, calls = self.runner(reduced=reduced, transform=transform)
            with (
                patch.dict(sys.modules, {module: helper}),
                self.assertRaises(RuntimeError),
            ):
                fn(obj, torch.ones(2, 4), torch.ones(2, 4), defer_tp_reduction=True)
            self.assertEqual(calls, [])


class MetadataTest(unittest.TestCase):
    def test_host_metadata_excludes_missing_mixed_speculative_and_non_8k(self):
        good = NS(
            num_prefills=1,
            num_prefill_tokens=8192,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
        )
        self.assertTrue(
            helper.pure_prefill_metadata({"a": good, "b": good}, ("a", "b"), 8192)
        )
        for field, value in [
            ("num_decodes", 1),
            ("num_decode_tokens", 1),
            ("num_spec_decodes", 1),
            ("num_spec_decode_tokens", 4),
            ("num_prefills", 0),
            ("num_prefill_tokens", 8191),
            ("num_prefills", torch.tensor(1)),
        ]:
            bad = NS(**vars(good))
            setattr(bad, field, value)
            self.assertFalse(
                helper.pure_prefill_metadata({"a": good, "b": bad}, ("a", "b"), 8192)
            )
        self.assertFalse(helper.pure_prefill_metadata({"a": good}, ("a", "b"), 8192))
        self.assertFalse(helper.pure_prefill_metadata([{"a": good}], ("a",), 8192))
        self.assertFalse(helper.pure_prefill_metadata({"a": good}, ("a",), 4096))

    def admission_stubs(self, *, comm_rank=0, comm_device="cuda:0", peer_vote=None):
        comm = NS(
            available=True,
            disabled=False,
            world_size=4,
            rank=comm_rank,
            device=torch.device(comm_device),
        )
        group = NS(
            world_size=4,
            rank_in_group=0,
            cpu_group=object(),
            device_communicator=NS(pynccl_comm=comm),
        )
        metadata = NS(
            num_prefills=1,
            num_prefill_tokens=8192,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
        )
        context = NS(
            cudagraph_runtime_mode=NS(name="NONE"),
            ubatch_slices=None,
            attn_metadata={"a": metadata},
        )
        modules = {
            "vllm.distributed": NS(get_tp_group=lambda: group),
            "vllm.forward_context": NS(
                get_forward_context=lambda: context,
                is_forward_context_available=lambda: True,
            ),
        }
        votes = []

        def vote(output, value, **kwargs):
            votes.append(value)
            output[:] = [value] * 4
            if peer_vote is not None:
                output[2] = peer_vote

        model = NS(_mhc_prefill_enabled=True)
        hidden = NS(
            shape=(8192, 4096),
            device=torch.device("cuda:0"),
            is_cuda=True,
            dtype=torch.bfloat16,
        )
        return modules, vote, votes, model, hidden

    def test_rank_device_errors_are_voted_before_any_owner_is_created(self):
        for rank, device in [(1, "cuda:0"), (0, "cuda:1")]:
            modules, vote, votes, model, hidden = self.admission_stubs(
                comm_rank=rank, comm_device=device
            )
            with (
                patch.dict(sys.modules, modules),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
                patch.object(torch.distributed, "all_gather_object", side_effect=vote),
                patch.object(helper, "validate_model", return_value=("a",)),
            ):
                with self.assertRaisesRegex(RuntimeError, "capability vote"):
                    helper.maybe_create(model, hidden, NS(shape=(8192,)))
            self.assertFalse(votes[0][0])
            self.assertIn("ownership mismatch", votes[0][1])

    def test_peer_ineligibility_falls_back_before_partial_ownership(self):
        modules, vote, votes, model, hidden = self.admission_stubs(
            peer_vote=(False, None)
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.distributed, "all_gather_object", side_effect=vote),
            patch.object(helper, "validate_model", return_value=("a",)),
        ):
            self.assertIsNone(helper.maybe_create(model, hidden, NS(shape=(8192,))))
        self.assertEqual(votes, [(True, None)])

    def test_startup_feature_disagreement_fails_on_disabled_rank_too(self):
        modules, vote, votes, model, hidden = self.admission_stubs(peer_vote=True)
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch.distributed, "all_gather_object", side_effect=vote),
        ):
            with self.assertRaisesRegex(RuntimeError, "All TP ranks"):
                helper.configure(model, NS(), False)
        self.assertEqual(votes, [False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
