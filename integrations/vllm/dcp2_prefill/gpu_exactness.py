"""Single-GPU component exactness; run under the workstation GPU lock.

Pass --sources with the image's exported site-packages tree. This executes
the actual B12X kernels, not a numerical substitute for them. It emulates
two TP row owners locally; it does not qualify inter-GPU collectives.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--kda-only', action='store_true', help='Test the admitted feature only')
    args = parser.parse_args()
    sys.path.insert(0, args.sources)
    import torch
    from b12x.preparation import PreparationSession, PreparedCall, FrozenMapping, require_prepared
    from b12x.sequence import kda_prefill as kda
    from b12x.norm import mhc
    from b12x.norm.mhc import _impl as mhc_impl
    from b12x.testing.mhc import make_inputs

    device = torch.device('cuda:0')
    print(json.dumps(dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                         cuda=torch.version.cuda, cutlass=importlib.metadata.version('nvidia-cutlass-dsl'),
                         sources=args.sources,
                         b12x_tree_sha256=hashlib.sha256(''.join(
                             str(p.relative_to(Path(args.sources))).replace('\\', '/')+':'+hashlib.sha256(p.read_bytes()).hexdigest()+'\n'
                             for p in sorted((Path(args.sources)/'b12x').rglob('*.py'))).encode()).hexdigest())), flush=True)
    results = []

    def equal(label, left, right):
        torch.cuda.synchronize()
        ok = torch.equal(left, right) and bool(torch.isfinite(left).all())
        row = dict(test=label, identical=ok, max_abs=float((left.float()-right.float()).abs().max()))
        results.append(row)
        print(json.dumps(row), flush=True)

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        def run_kda(inputs, pool, start, end, capacity, targets=()):
            n = end-start
            ints = lambda x: torch.tensor(x, device=device, dtype=torch.int32)  # noqa: E731
            tensors = {key: value[start:end].contiguous() for key, value in inputs.items()
                       if key in ('q', 'k', 'v', 'raw_g', 'raw_beta')}
            tensors.update(A_log=inputs['A_log'], dt_bias=inputs['dt_bias'],
                           recurrent_state=pool, cu_seqlens=ints([0, n]),
                           initial_state_indices=ints([0]), final_state_indices=ints([1]),
                           checkpoint_state_indices=ints([[2+i for i in range(len(targets))]
                                                         + [-1]*(capacity-len(targets))]),
                           checkpoint_offsets=ints([list(targets)+[0]*(capacity-len(targets))]),
                           num_seqs=ints([1]), num_tokens=ints([n]),
                           output=torch.empty_like(tensors['v']))
            if capacity == 1:
                for key in ('checkpoint_state_indices', 'checkpoint_offsets'):
                    tensors[key] = tensors[key].flatten()
            caps = kda.Caps(device=device, max_tokens=n, max_seqs=1,
                            max_state_slots=6, heads=32, checkpoint_export=True,
                            max_checkpoints=capacity)
            plan = kda.plan(caps, invocation=kda.invocation_from_tensors(**tensors))
            def trial(state):
                spec, = state.layout.scratch_specs()
                binding = state.bind(scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **tensors)
                saved = pool.clone()
                reset = lambda: pool.copy_(saved)  # noqa: E731
                return PreparedCall(run=lambda: state.run(binding, lower_bound=-5.0),
                                    output=binding.output, reset=reset, restore=reset)
            session.prepare((plan.request(name='dcp2-kda', prepare_call=trial),))
            state = require_prepared(plan, 'sequence.kda_prefill')
            spec, = state.layout.scratch_specs()
            bound = kda.bind(plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **tensors)
            kda.run(bound, lower_bound=-5.0)
            assert bound.error_code.item() == 0
            return bound.output

        # GLM has 64 KDA heads, hence 32 per TP2 rank, with full query rows
        # at DCP2. Compare one 8192-token call against 2048-token boundaries.
        for seed in (13, 29):
            torch.manual_seed(seed)
            inputs = {key: (torch.randn(8192, 32, 128, device=device)*0.25).bfloat16()
                      for key in ('q', 'k', 'v', 'raw_g')}
            inputs.update(raw_beta=torch.randn(8192, 32, device=device).bfloat16(),
                          A_log=torch.randn(32, device=device)*0.1,
                          dt_bias=torch.randn(32, 128, device=device)*0.1)
            initial = torch.randn(6, 32, 128, 128, device=device)*0.1
            for cold in (True, False):
                if cold:
                    initial[0].zero_()
                else:
                    initial[0].normal_(std=0.1)
                coalesced = initial.clone()
                whole = run_kda(inputs, coalesced, 0, 8192, 4, (2048, 4096, 6144))
                split = initial.clone()
                pieces, states = [], []
                for start in range(0, 8192, 2048):
                    pieces.append(run_kda(inputs, split, start, start+2048, 1).clone())
                    states.append(split[1].clone())
                    split[0].copy_(split[1])
                prefix = f'kda-seed{seed}-cold{cold}'
                equal(prefix+'-output', whole, torch.cat(pieces))
                for i, expected in enumerate(states):
                    equal(prefix+f'-state{i+1}', coalesced[2+i if i<3 else 1], expected)

        if args.kda_only:
            assert all(row['identical'] for row in results), 'KDA exactness failed'
            print(json.dumps(dict(passed=len(results), scope='KDA kernels; no distributed serving')), flush=True)
            return

        def run_mhc(operation, operands, options, capacity):
            residual = operands[0 if operation == 'pre' else 1]
            invocation = FrozenMapping(dict(operation=operation, output_mode='functional',
                expanded_residual=operation == 'pre' and residual.ndim == 3,
                has_norm_weight=options.get('norm_weight') is not None,
                norm_weight_dtype='bfloat16', has_fn_bf16=options.get('fn_bf16') is not None,
                rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-6))
            plan = mhc.plan(mhc.Caps(device=device, max_tokens=capacity, hidden_size=4096),
                            invocation=invocation)
            entry = getattr(mhc_impl, '_b12x_mhc_'+operation+'_impl')
            session.prepare((plan.request(name='dcp2-mhc-'+operation,
                prepare_call=lambda state: PreparedCall(run=lambda: entry(*operands, **options, _state=state))),))
            return getattr(mhc, 'run_'+operation)(*operands, **options, plan=plan)

        for rows in (4096, 8192):
            residual, x, fn, scale, bias = make_inputs(tokens=rows, hidden_size=4096, seed=rows, device=device)
            norm = torch.ones(4096, dtype=torch.bfloat16, device=device)
            options = dict(rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-6, norm_weight=norm)
            pre = run_mhc('pre', (residual, fn, scale, bias), options, rows)
            # The B12X residual API returns residual, post, comb, y.
            r, post, comb, y = pre
            for operation, operands, opts in (
                ('pre', (residual, fn, scale, bias), options),
                ('post_pre', (x, r, post, comb, fn, scale, bias), options),
            ):
                full = run_mhc(operation, operands, opts, rows)
                halves = []
                row_operands = 1 if operation == 'pre' else 4
                for rank in range(2):
                    owned = tuple(t[rank*rows//2:(rank+1)*rows//2].contiguous() if i<row_operands else t
                                  for i, t in enumerate(operands))
                    # vLLM B12xMHCResidual._plan_for falls back to its full
                    # ceiling plan: only graph/compile sizes have exact plans.
                    halves.append(run_mhc(operation, owned, opts, rows))
                for i, value in enumerate(full):
                    equal(f'mhc-{operation}-{rows}-tensor{i}', value, torch.cat([h[i] for h in halves]))
            full = run_mhc('post', (x, r, post, comb), {}, rows)
            owned = [run_mhc('post', tuple(t[rank*rows//2:(rank+1)*rows//2].contiguous()
                                          for t in (x, r, post, comb)), {}, rows) for rank in range(2)]
            equal(f'mhc-post-{rows}', full, torch.cat(owned))
            equal(f'mhc-contract-{rows}', full.mean(1), torch.cat(owned).mean(1))
    assert all(row['identical'] for row in results), 'component exactness failed'
    print(json.dumps(dict(passed=len(results), scope='single GPU kernels; no distributed serving')), flush=True)


if __name__ == '__main__':
    main()
