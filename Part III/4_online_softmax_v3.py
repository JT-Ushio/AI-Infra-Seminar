import triton
from triton import language as tl
import torch
from torch.profiler import profile, ProfilerActivity


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit
def softmax_kernel_online_v1(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    m = tl.full((), value=-float("inf"), dtype=output_ptr.dtype.element_ty)
    z = tl.full((), value=0, dtype=output_ptr.dtype.element_ty)
    before_last = prev_multiple_of(N, TILE_N)
    for start_cta in range(0, before_last, TILE_N):
        offsets_cta = start_cta + tl.arange(0, TILE_N)
        offsets_x = pid_m * N + offsets_cta
        x = tl.load(input_ptr + offsets_x)
        # x = tl.load(input_ptr + offsets_x, mask=offsets_cta < N, other=-float("inf"))
        new_m = tl.maximum(m, tl.max(x))
        new_z = tl.exp(m - new_m) * z + tl.sum(tl.exp(x - new_m))
        m = new_m
        z = new_z

    for start_cta in range(before_last, N, TILE_N):
        offsets_cta = start_cta + tl.arange(0, TILE_N)
        offsets_x = pid_m * N + offsets_cta
        x = tl.load(input_ptr + offsets_x, mask=offsets_cta < N, other=-float("inf"))
        new_m = tl.maximum(m, tl.max(x))
        new_z = tl.exp(m - new_m) * z + tl.sum(tl.exp(x - new_m))
        m = new_m
        z = new_z

    for start_cta in range(0, TILE_N, TILE_N):
        offsets_cta = before_last + tl.arange(0, TILE_N)
        offsets_x = pid_m * N + offsets_cta
        x = tl.load(input_ptr + offsets_x, mask=offsets_cta < N, other=-float("inf"))
        e = tl.exp(x - m)
        out = e / z
        tl.store(output_ptr + offsets_x, out, mask=offsets_cta < N)

    for start_cta in range(TILE_N, N, TILE_N):
        offsets_cta = (before_last - start_cta) + tl.arange(0, TILE_N)
        offsets_x = pid_m * N + offsets_cta
        x = tl.load(input_ptr + offsets_x)
        e = tl.exp(x - m)
        out = e / z
        tl.store(output_ptr + offsets_x, out)


def softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = min(4096, triton.next_power_of_2(N))
    grid = (M, 1, 1)
    softmax_kernel_online_v1[grid](out, x, M, N, TILE_N)
    return out


# x = torch.randn((4096, 32768), device="cuda")
# x = torch.randn((4096, 8192), device="cuda")
x = torch.randn((4096, 4096), device="cuda")

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
assert torch.allclose(out1, out0, atol=1e-5)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
