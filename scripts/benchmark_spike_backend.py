import argparse
import time

import torch

try:
    from spikingjelly.activation_based import functional, neuron
except ImportError as exc:
    raise ImportError("spikingjelly is required to run this benchmark.") from exc


def benchmark_backend(backend, t_steps, batch_size, features, warmup, iters):
    if backend != "torch" and not torch.cuda.is_available():
        return None, "CUDA unavailable"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        node = neuron.LIFNode(step_mode="m", backend=backend).to(device)
        x = torch.randn(t_steps, batch_size, features, device=device)

        for _ in range(warmup):
            functional.reset_net(node)
            _ = node(x)
        if device == "cuda":
            torch.cuda.synchronize()

        st = time.perf_counter()
        for _ in range(iters):
            functional.reset_net(node)
            _ = node(x)
        if device == "cuda":
            torch.cuda.synchronize()
        et = time.perf_counter()
        return (et - st) / iters, None
    except Exception as exc:
        return None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Benchmark spikingjelly backend latency.")
    parser.add_argument("--t-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["torch", "cupy", "triton"],
        help="Backends to test.",
    )
    args = parser.parse_args()

    print(
        f"Benchmark config: T={args.t_steps}, B={args.batch_size}, C={args.features}, "
        f"warmup={args.warmup}, iters={args.iters}, cuda={torch.cuda.is_available()}"
    )

    results = {}
    for backend in args.backends:
        latency, error = benchmark_backend(
            backend,
            args.t_steps,
            args.batch_size,
            args.features,
            args.warmup,
            args.iters,
        )
        if error is not None:
            print(f"[{backend}] unavailable: {error}")
        else:
            results[backend] = latency
            print(f"[{backend}] avg latency: {latency * 1000:.3f} ms")

    if "torch" in results:
        base = results["torch"]
        for backend, latency in results.items():
            if backend == "torch":
                continue
            speedup = base / latency if latency > 0 else 0.0
            print(f"speedup torch->{backend}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
