"""
Discrete Information-Theoretic Analysis for LAPA delta tokens
============================================================

Input:
    z_rgb_val.jsonl
    z_depth_val.jsonl

Each line should contain a field like:
    {
        "image": "/img0001.png",
        "delta": ["5", "5", "5", "2"],
        "instruction": "..."
    }

This script compares delta_rgb and delta_depth.

Questions:
    Q1. Does depth delta add new information beyond RGB delta?
        -> H(delta_depth | delta_rgb)

    Q2. How much information is redundant/shared?
        -> I(delta_depth; delta_rgb)

Usage:
    python latent_info_analysis.py \
        --rgb_jsonl z_rgb_val.jsonl \
        --depth_jsonl z_depth_val.jsonl \
        --field delta \
        --num_samples 1000

If both files have image field and the order may be different:
    python latent_info_analysis.py \
        --rgb_jsonl z_rgb_val.jsonl \
        --depth_jsonl z_depth_val.jsonl \
        --field delta \
        --key_field image \
        --num_samples 1000
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────
#  JSONL loading
# ─────────────────────────────────────────────────────────

def load_jsonl_delta(path, field="delta", key_field=None):
    """
    Load delta vectors from a JSONL file.

    Parameters
    ----------
    path : str or Path
        Path to JSONL file.
    field : str
        Field name containing delta tokens.
    key_field : str or None
        Optional field used for alignment, e.g. "image".

    Returns
    -------
    keys : list[str] or None
        Keys used for alignment. None if key_field is None.
    deltas : np.ndarray
        Shape: (N, code_seq_len), dtype int64
    """
    path = Path(path)

    keys = []
    deltas = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_id} in {path}. "
                    f"Original error: {e}"
                )

            if field not in obj:
                raise KeyError(
                    f"Missing field '{field}' at line {line_id} in {path}. "
                    f"Available keys: {list(obj.keys())}"
                )

            delta = obj[field]

            # Support both list and string-encoded list
            if isinstance(delta, str):
                try:
                    delta = json.loads(delta)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Field '{field}' at line {line_id} is a string "
                        f"but not a valid JSON list: {delta}"
                    )

            if not isinstance(delta, (list, tuple)):
                raise TypeError(
                    f"Field '{field}' at line {line_id} must be list/tuple, "
                    f"got {type(delta)}"
                )

            try:
                delta = [int(x) for x in delta]
            except Exception as e:
                raise ValueError(
                    f"Cannot convert delta to int list at line {line_id}: {delta}. "
                    f"Original error: {e}"
                )

            deltas.append(delta)

            if key_field is not None:
                if key_field not in obj:
                    raise KeyError(
                        f"Missing key_field '{key_field}' at line {line_id} in {path}. "
                        f"Available keys: {list(obj.keys())}"
                    )
                keys.append(str(obj[key_field]))

    if len(deltas) == 0:
        raise ValueError(f"No valid samples found in {path}")

    deltas = np.asarray(deltas, dtype=np.int64)

    if deltas.ndim != 2:
        raise ValueError(
            f"Expected delta array shape (N, code_seq_len), got {deltas.shape}"
        )

    if key_field is None:
        return None, deltas

    return keys, deltas


def align_by_key(rgb_keys, rgb_delta, depth_keys, depth_delta):
    """
    Align RGB and depth samples using a shared key, e.g. image path.
    """
    rgb_map = {k: v for k, v in zip(rgb_keys, rgb_delta)}
    depth_map = {k: v for k, v in zip(depth_keys, depth_delta)}

    common_keys = sorted(set(rgb_map.keys()) & set(depth_map.keys()))

    if len(common_keys) == 0:
        raise ValueError("No common keys found between RGB and depth files.")

    z_rgb = np.asarray([rgb_map[k] for k in common_keys], dtype=np.int64)
    z_depth = np.asarray([depth_map[k] for k in common_keys], dtype=np.int64)

    return z_rgb, z_depth, common_keys


# ─────────────────────────────────────────────────────────
#  Discrete entropy and mutual information
# ─────────────────────────────────────────────────────────

def entropy_from_counter(counter, total):
    """
    Discrete entropy in nats.
    """
    probs = np.asarray(list(counter.values()), dtype=np.float64) / float(total)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def entropy_discrete(Z):
    """
    Entropy H(Z), treating each row as one discrete tuple.

    Example:
        [5, 5, 5, 2] -> state (5, 5, 5, 2)
    """
    tuples = [tuple(row.tolist()) for row in Z]
    counter = Counter(tuples)
    return entropy_from_counter(counter, len(tuples)), counter


def joint_entropy_discrete(X, Y):
    """
    Joint entropy H(X,Y), treating each pair of rows as one joint state.
    """
    joint_states = [
        (tuple(x.tolist()), tuple(y.tolist()))
        for x, y in zip(X, Y)
    ]
    counter = Counter(joint_states)
    return entropy_from_counter(counter, len(joint_states)), counter


def per_position_match_rate(z_rgb, z_depth):
    """
    Token-level match rate for each delta position.
    """
    return np.mean(z_rgb == z_depth, axis=0)


def full_delta_match_rate(z_rgb, z_depth):
    """
    Full sequence match rate.
    True if the whole delta vector is identical.
    """
    return float(np.mean(np.all(z_rgb == z_depth, axis=1)))


def unique_state_count(Z):
    """
    Number of unique delta states.
    """
    return len({tuple(row.tolist()) for row in Z})


# ─────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────

@dataclass
class DiscreteInfoAnalysis:
    H_rgb: float
    H_depth: float
    H_joint: float
    n_samples: int
    rgb_unique: int
    depth_unique: int
    joint_unique: int
    full_match_rate: float
    token_match_rate: np.ndarray

    @property
    def MI(self):
        """
        I(depth; rgb) = H(rgb) + H(depth) - H(rgb, depth)
        """
        return max(self.H_rgb + self.H_depth - self.H_joint, 0.0)

    @property
    def H_depth_given_rgb(self):
        """
        H(depth | rgb) = H(rgb, depth) - H(rgb)
        This is the new information depth adds beyond RGB.
        """
        return max(self.H_joint - self.H_rgb, 0.0)

    @property
    def H_rgb_given_depth(self):
        """
        H(rgb | depth) = H(rgb, depth) - H(depth)
        """
        return max(self.H_joint - self.H_depth, 0.0)

    @property
    def info_gain(self):
        """
        H(concat) - H(rgb), where concat = (rgb, depth).
        """
        return self.H_depth_given_rgb

    @property
    def redundancy_ratio(self):
        """
        Fraction of depth information that overlaps with RGB.
        """
        if self.H_depth <= 1e-12:
            return 0.0
        return min(self.MI / self.H_depth, 1.0)

    @property
    def novelty_ratio(self):
        """
        Fraction of depth information that is new given RGB.
        """
        return 1.0 - self.redundancy_ratio

    @property
    def relative_gain(self):
        """
        Information gain normalized by H(rgb).
        """
        if self.H_rgb <= 1e-12:
            return 0.0
        return self.info_gain / self.H_rgb

    def verdict_redundancy(self):
        r = self.redundancy_ratio
        if r > 0.75:
            return "HIGH: depth mostly duplicates RGB"
        if r > 0.40:
            return "MODERATE: substantial overlap"
        if r > 0.15:
            return "LOW: mostly complementary"
        return "NEGLIGIBLE: nearly independent"

    def verdict_gain(self):
        g = self.relative_gain
        if g > 0.40:
            return "LARGE: concat is substantially richer than RGB alone"
        if g > 0.15:
            return "MODERATE: meaningful gain from depth"
        if g > 0.05:
            return "SMALL: marginal gain from depth"
        return "NEGLIGIBLE: concat is close to RGB alone"

    def print_report(self):
        w = 72
        sep = "-" * w

        print("\n" + "=" * w)
        print("Discrete Information Analysis: delta_depth vs delta_rgb")
        print("=" * w)

        print("\nBasic statistics")
        print(f"  Number of samples                 = {self.n_samples}")
        print(f"  Unique delta_rgb states            = {self.rgb_unique}")
        print(f"  Unique delta_depth states          = {self.depth_unique}")
        print(f"  Unique joint states                = {self.joint_unique}")
        print(f"  Full delta match rate              = {self.full_match_rate:>8.2%}")

        print("\nToken match rate by position")
        for i, rate in enumerate(self.token_match_rate):
            print(f"  Position {i:<2d}                       = {rate:>8.2%}")

        print("\n" + sep)
        print("Marginal entropies")
        print(sep)
        print(f"  H(delta_rgb)                      = {self.H_rgb:>8.4f} nats")
        print(f"  H(delta_depth)                    = {self.H_depth:>8.4f} nats")

        print("\n" + sep)
        print("Shared / redundant information")
        print(sep)
        print(f"  I(delta_depth; delta_rgb)         = {self.MI:>8.4f} nats")
        print(f"  Redundancy ratio                  = {self.redundancy_ratio:>8.2%} of H(delta_depth)")
        print(f"  Verdict                           = {self.verdict_redundancy()}")

        print("\n" + sep)
        print("Unique / novel information")
        print(sep)
        print(f"  H(delta_depth | delta_rgb)        = {self.H_depth_given_rgb:>8.4f} nats")
        print(f"  H(delta_rgb | delta_depth)        = {self.H_rgb_given_depth:>8.4f} nats")
        print(f"  Novelty ratio                     = {self.novelty_ratio:>8.2%} of H(delta_depth)")

        print("\n" + sep)
        print("Does concat(depth, RGB) add information over RGB?")
        print(sep)
        print(f"  H(delta_rgb)                      = {self.H_rgb:>8.4f} nats")
        print(f"  H(delta_depth, delta_rgb)         = {self.H_joint:>8.4f} nats")
        print(f"  Information gain                  = {self.info_gain:>8.4f} nats")
        print(f"  Relative gain                     = {self.relative_gain:>8.2%}")
        print(f"  Verdict                           = {self.verdict_gain()}")

        print("=" * w + "\n")


def analyse_discrete_delta(z_rgb, z_depth):
    """
    Run discrete information analysis on RGB and depth delta arrays.

    Parameters
    ----------
    z_rgb : np.ndarray
        Shape: (N, code_seq_len)
    z_depth : np.ndarray
        Shape: (N, code_seq_len)
    """
    if z_rgb.shape[0] != z_depth.shape[0]:
        raise ValueError(
            f"Need same number of samples. "
            f"Got z_rgb={z_rgb.shape[0]}, z_depth={z_depth.shape[0]}"
        )

    if z_rgb.shape[1] != z_depth.shape[1]:
        raise ValueError(
            f"Need same delta length. "
            f"Got z_rgb={z_rgb.shape[1]}, z_depth={z_depth.shape[1]}"
        )

    H_rgb, rgb_counter = entropy_discrete(z_rgb)
    H_depth, depth_counter = entropy_discrete(z_depth)
    H_joint, joint_counter = joint_entropy_discrete(z_rgb, z_depth)

    result = DiscreteInfoAnalysis(
        H_rgb=H_rgb,
        H_depth=H_depth,
        H_joint=H_joint,
        n_samples=z_rgb.shape[0],
        rgb_unique=len(rgb_counter),
        depth_unique=len(depth_counter),
        joint_unique=len(joint_counter),
        full_match_rate=full_delta_match_rate(z_rgb, z_depth),
        token_match_rate=per_position_match_rate(z_rgb, z_depth),
    )

    result.print_report()
    return result


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare z_rgb_val.jsonl and z_depth_val.jsonl using discrete delta tokens."
    )

    parser.add_argument(
        "--rgb_jsonl",
        required=True,
        help="Path to z_rgb_val.jsonl",
    )

    parser.add_argument(
        "--depth_jsonl",
        required=True,
        help="Path to z_depth_val.jsonl",
    )

    parser.add_argument(
        "--field",
        default="delta",
        help="Field name to read from each JSONL line. Default: delta",
    )

    parser.add_argument(
        "--key_field",
        default=None,
        help=(
            "Optional key for alignment, e.g. image. "
            "If not set, samples are aligned by line order."
        ),
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples/lines to compare. Default: use all samples.",
    )

    args = parser.parse_args()

    rgb_keys, z_rgb = load_jsonl_delta(
        args.rgb_jsonl,
        field=args.field,
        key_field=args.key_field,
    )

    depth_keys, z_depth = load_jsonl_delta(
        args.depth_jsonl,
        field=args.field,
        key_field=args.key_field,
    )

    if args.key_field is not None:
        z_rgb, z_depth, common_keys = align_by_key(
            rgb_keys=rgb_keys,
            rgb_delta=z_rgb,
            depth_keys=depth_keys,
            depth_delta=z_depth,
        )

        print(f"Aligned by key_field='{args.key_field}'")
        print(f"Common samples before selection: {len(common_keys)}")

    else:
        min_n = min(z_rgb.shape[0], z_depth.shape[0])

        if z_rgb.shape[0] != z_depth.shape[0]:
            print(
                f"Warning: files have different numbers of samples. "
                f"RGB={z_rgb.shape[0]}, depth={z_depth.shape[0]}. "
                f"Using first {min_n} samples."
            )

        z_rgb = z_rgb[:min_n]
        z_depth = z_depth[:min_n]
        common_keys = None

        print("Aligned by line order")
        print(f"Samples before selection: {min_n}")

    total_n = z_rgb.shape[0]

    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError("--num_samples must be positive.")

        num_samples = min(args.num_samples, total_n)

        z_rgb = z_rgb[:num_samples]
        z_depth = z_depth[:num_samples]

        if common_keys is not None:
            common_keys = common_keys[:num_samples]

        print(f"Selected samples: {num_samples}/{total_n}")
    else:
        print(f"Using all samples: {total_n}")

    print(f"z_rgb shape             : {z_rgb.shape}")
    print(f"z_depth shape           : {z_depth.shape}")
    print(f"First RGB delta         : {z_rgb[0].tolist()}")
    print(f"First depth delta       : {z_depth[0].tolist()}")

    analyse_discrete_delta(z_rgb, z_depth)


if __name__ == "__main__":
    main()