import triton
from triton import language as tl
import torch
from torch.profiler import profile, ProfilerActivity


@triton.jit
def softmax_kernel(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    pid_m = tl.program_id(0)  # (0...M-1, 0, 0)
    n_offsets = tl.arange(0, TILE_N)  # [0, ..., TILE_N-1]
    offset = pid_m * N + n_offsets  # (pid_m, [0, ..., TILE_N-1])

    x = tl.load(input_ptr + offset, mask=n_offsets < N, other=-float("inf"))
    m = tl.max(x)
    e = tl.exp(x - m)
    z = tl.sum(e)
    out = e / z
    tl.store(output_ptr + offset, out, mask=n_offsets < N)


def softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = triton.next_power_of_2(N)
    grid = (M, 1, 1)
    softmax_kernel[grid](out, x, M, N, TILE_N)
    return out


x = torch.randn((4096, 32768), device="cuda")
# x = torch.randn((4096, 8192), device="cuda")
# x = torch.randn((4096, 4096), device="cuda")

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/naive_softmax"),
    record_shapes=False,
    with_stack=False,
) as prof:
    for i in range(5):
        with torch.no_grad():
            out1 = softmax(x)
        with torch.no_grad():
            out0 = torch.softmax(x, dim=-1)
        prof.step()

out1 = softmax(x)
out0 = torch.softmax(x, dim=-1)
assert torch.allclose(out1, out0, atol=1e-3)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
