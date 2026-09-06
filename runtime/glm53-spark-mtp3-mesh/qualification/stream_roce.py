"""Validate bundled RoCEnante staging across CUDA streams with exact BF16 sums."""
import argparse
import json
import torch
import torch.distributed as dist

parser=argparse.ArgumentParser()
parser.add_argument('--bytes',type=int,required=True)
args,_=parser.parse_known_args()
dist.init_process_group('gloo')
rank=dist.get_rank()
assert dist.get_world_size()==4
torch.cuda.set_device(0)
import b12x.comm  # noqa: E402 -- device selection precedes transport import
b12x.comm.__path__.insert(0,'/opt/spark-sircl/b12x_overlay/b12x/comm')
from b12x.comm import roce  # noqa: E402 -- resolve only the image-bundled transport
runtime=roce.AllReduce.from_exchange_group(exchange_group=dist.group.WORLD,device=torch.device('cuda',0),
                                         max_size=2<<20,max_gather_bytes=2<<20)
runtime.prepare((torch.bfloat16,))
numel=args.bytes//2
streams=[torch.cuda.Stream(),torch.cuda.Stream()]
inputs=[torch.empty(numel+1,device='cuda',dtype=torch.bfloat16)[1:] for _ in range(16)]
outputs=[torch.empty_like(inp.new_empty(numel+1))[1:] for inp in inputs]
assert all(t.data_ptr()%16 for t in inputs+outputs)
# Establish shared scratch before exercising asynchronous alternating callers.
inputs[0].fill_(rank+1)
runtime.all_reduce(inputs[0],out=outputs[0])
torch.cuda.synchronize()
dist.barrier()
for i,(inp,out) in enumerate(zip(inputs,outputs)):
    with torch.cuda.stream(streams[i%2]):
        inp.fill_(rank+1+i)
        out.fill_(-123)
        runtime.all_reduce(inp,out=out)
torch.cuda.synchronize()
runtime.check_health()
assert all(torch.equal(out,torch.full_like(out,10+4*i)) for i,out in enumerate(outputs))
dist.barrier()
graph=torch.cuda.CUDAGraph()
streams[0].wait_stream(torch.cuda.current_stream())
with torch.cuda.graph(graph,stream=streams[0]):
    runtime.all_reduce(inputs[0],out=outputs[0])
for i in (20,21):
    inputs[0].fill_(rank+1+i)
    outputs[0].fill_(-321)
    graph.replay()
    torch.cuda.synchronize()
    runtime.check_health()
    assert torch.equal(outputs[0],torch.full_like(outputs[0],10+4*i))
dist.barrier()
# A separate Python capture context must not admit another stream in one CUDA capture.
probe=torch.zeros(1,device='cuda')
probe.add_(1)
torch.cuda.synchronize()
guard_graph=torch.cuda.CUDAGraph()
rejected=False
with torch.cuda.graph(guard_graph,stream=streams[0]):
    probe.add_(1)
    with runtime.capture():
        runtime._order_stream(True)
    streams[1].wait_stream(streams[0])
    with torch.cuda.stream(streams[1]):
        with runtime.capture():
            try:
                runtime._order_stream(True)
            except RuntimeError as error:
                assert 'one stream' in str(error)
                rejected=True
    streams[0].wait_stream(streams[1])
assert rejected
torch.cuda.synchronize()
runtime.check_health()
record={'rank':rank,'payload_bytes':args.bytes,'alternating_stream_calls':16,
        'misaligned_input_output':True,'changed_input_graph_replays':2,
        'capture_stream_rejected':rejected,'passed':True}
rows=[None]*4
dist.all_gather_object(rows,record)
if rank==0:
    print('EVIDENCE_JSON '+json.dumps({'checks':rows,'passed':all(row['passed'] for row in rows)}),flush=True)
runtime.close()
dist.destroy_process_group()
