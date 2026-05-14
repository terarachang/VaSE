import argparse
import os
import torch
from tqdm import tqdm
from hqq.core.quantize import Quantizer


def compute_layer_correlation(layer_dir, n_examples, n_bits, group_size, n_sink, device):
    se_chunks, range_chunks = [], []
    d = None

    for i in range(n_examples):
        path = os.path.join(layer_dir, f"ex-{i}.pt")
        x = torch.load(path, map_location=device, weights_only=False)
        x = x.to(torch.float32)             # HQQ optimizer is happier in fp32
        x = x[:, n_sink:]                   # drop sink tokens from analysis

        if d is None:
            d = x.shape[-1]
            assert d % group_size == 0, \
                f"d ({d}) must be divisible by group_size ({group_size})"

        # HQQ's Quantizer expects 2D. Flatten leading dims so the last dim
        # becomes axis=1, then group along axis=1 (the original last dim).
        x_2d = x.reshape(-1, d)

        # HQQ is inherently asymmetric: it learns both `scale` and `zero`
        # via its half-quadratic solver, with W_q = round(W * scale + zero).
        W_q, meta = Quantizer.quantize(
            x_2d,
            nbits=n_bits,
            group_size=group_size,
            axis=1,
            channel_wise=True,
            optimize=True,
            round_zero=False,
            bitpack=False,
            device=device,
        )
        x_dq = Quantizer.dequantize(W_q, meta).reshape(x.shape)

        se_chunks.append((x - x_dq).pow(2).mean(-1).flatten()) # per-token squared error
        range_chunks.append((x.amax(-1) - x.amin(-1)).flatten())

    se_flat = torch.cat(se_chunks)
    range_flat = torch.cat(range_chunks)
    return torch.corrcoef(torch.stack([se_flat, range_flat]))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--vcache_root", type=str, default="vcaches")
    parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument("--n_bits", type=int, required=True)
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    layers = sorted(
        [name for name in os.listdir(args.vcache_root) if name.startswith("L")],
        key=lambda s: int(s[1:]),
    )

    corrs = []
    for layer in tqdm(layers, desc="layers"):
        c = compute_layer_correlation(
            os.path.join(args.vcache_root, layer),
            args.n_examples, args.n_bits, args.group_size, args.n_sink, args.device,
        )
        corrs.append(c)
        tqdm.write(f"{layer}: corr={c:.4f}")

    corrs_t = torch.tensor(corrs)
    print(f"layers              : {layers}")
    print(f"n_bits / group_size : {args.n_bits} / {args.group_size}")
    print(f"corrs               : {corrs_t}")

    out_dir = "corr_se_range"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"b{args.n_bits}g{args.group_size}.pt")
    torch.save(corrs_t, out_path)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
