#!/usr/bin/env python3
"""
Analyze completions.jsonl files for generation quality problems.

Detected problem types:
  1. Garbage output    -- last 100 chars are >30% non-ASCII or >50% a single char
  2. Phrase loop       -- a short phrase repeats 5+ times in the last 500 chars
  3. Truncated         -- completion never closes </think> (hit token budget mid-reasoning)
  4. (Healthy)         -- has </think> and no loops/garbage

Usage — single files:
  python analyze_completions.py path/to/completions.jsonl [--examples N]
  python analyze_completions.py file1.jsonl file2.jsonl

Usage — method directories (auto-discovers run_*/completions.jsonl and averages):
  python analyze_completions.py --methods dir1 dir2 dir3
  python analyze_completions.py --methods gsm8k/Qwen3-4B/Evict/evict_cur/512_per_head \\
                                           gsm8k/Qwen3-4B/Evict/evict_attn_sample/512_per_head_smooth

  A "method directory" is any directory that contains run_0/, run_1/, … subdirectories
  each holding a completions.jsonl file.
"""

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── detection helpers ────────────────────────────────────────────────────────

def _is_garbage(comp: str) -> bool:
    """Last 100 chars are mostly non-ASCII (emoji floods) or a single repeated char."""
    tail = comp[-100:]
    if not tail:
        return False
    non_ascii = sum(1 for c in tail if ord(c) > 127)
    if non_ascii > 30:
        return True
    char_counts = {}
    for c in comp[-200:] if len(comp) >= 200 else comp:
        char_counts[c] = char_counts.get(c, 0) + 1
    if char_counts and max(char_counts.values()) / max(len(comp[-200:]), 1) > 0.5:
        return True
    return False


def _find_phrase_loop(comp: str) -> Optional[re.Match]:
    """Return the first phrase-loop match in the last 500 chars, or None."""
    tail = comp[-500:]
    for m in re.finditer(r'(.{10,40})\1{4,}', tail):
        return m
    return None


def _is_truncated(comp: str) -> bool:
    return '</think>' not in comp


# ── per-completion classification ────────────────────────────────────────────

@dataclass
class Result:
    index: int
    length: int
    label: str          # 'garbage' | 'phrase_loop' | 'truncated' | 'ok'
    detail: str = ''
    snippet: str = ''


def classify(index: int, comp: str) -> Result:
    length = len(comp)

    if _is_garbage(comp):
        return Result(index, length, 'garbage',
                      detail='Non-ASCII flood or single-char repetition in tail',
                      snippet=repr(comp[-200:][:120]))

    loop_match = _find_phrase_loop(comp)
    if loop_match:
        phrase = loop_match.group(1)
        count = len(re.findall(re.escape(phrase), comp))
        if re.search(r'[Ww]ait', phrase):
            sub = 'arithmetic-doubt loop'
        elif re.search(r'\\\\?boxed\{', phrase):
            sub = 'boxed-answer loop'
        elif re.search(r'[\U00010000-\U0010ffff]|\u2600-\u27BF', phrase):
            sub = 'emoji-decoration loop'
        else:
            sub = 'punctuation/markdown loop'
        return Result(index, length, 'phrase_loop',
                      detail=f'{sub} — phrase repeated ~{count}x: {repr(phrase[:60])}',
                      snippet=repr(comp[comp.rfind(phrase[:20]):comp.rfind(phrase[:20])+200]))

    if _is_truncated(comp):
        return Result(index, length, 'truncated',
                      detail='No </think> — hit token budget mid-reasoning',
                      snippet=repr(comp[-150:]))

    return Result(index, length, 'ok')


# ── file-level analysis ───────────────────────────────────────────────────────

LABELS = ['ok', 'truncated', 'phrase_loop', 'garbage']
LABEL_DISPLAY = {
    'ok':          'Complete (ok)',
    'truncated':   'Truncated',
    'phrase_loop': 'Phrase loop',
    'garbage':     'Garbage',
}


@dataclass
class FileStats:
    path: str
    total: int = 0
    counts: dict = field(default_factory=lambda: {k: 0 for k in LABELS})
    lengths: list = field(default_factory=list)
    bad_results: list = field(default_factory=list)

    def record(self, r: Result):
        self.total += 1
        self.lengths.append(r.length)
        self.counts[r.label] += 1
        if r.label != 'ok':
            self.bad_results.append(r)

    def pct(self, label):
        return self.counts[label] / self.total * 100 if self.total else 0

    def print_summary(self, show_examples: int = 2):
        print(f"File : {self.path}")
        print(f"Total: {self.total}")
        if self.lengths:
            print(f"Length  min:{min(self.lengths)}  max:{max(self.lengths)}  "
                  f"median:{statistics.median(self.lengths):.0f}  mean:{statistics.mean(self.lengths):.0f}")
        print()
        for key in LABELS:
            n = self.counts[key]
            print(f"  {LABEL_DISPLAY[key]:<28s} {n:4d}  ({self.pct(key):5.1f}%)")
        print()

        if show_examples and self.bad_results:
            by_label = {}
            for r in self.bad_results:
                by_label.setdefault(r.label, []).append(r)
            for label, results in by_label.items():
                print(f"  ── {label} examples ──")
                for r in results[:show_examples]:
                    print(f"    Sample {r.index} (len={r.length}): {r.detail}")
                    print(f"    {r.snippet}")
                    print()


def analyze_file(path: str) -> FileStats:
    stats = FileStats(path=path)
    with open(path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            comp = item.get('completion', '')
            stats.record(classify(i, comp))
    return stats


# ── method-level analysis (average over runs) ─────────────────────────────────

@dataclass
class MethodStats:
    """Aggregated statistics over all runs of a single method."""
    name: str                       # human-readable label
    run_stats: list = field(default_factory=list)   # List[FileStats]

    def add(self, fs: FileStats):
        self.run_stats.append(fs)

    @property
    def n_runs(self):
        return len(self.run_stats)

    def avg_pct(self, label) -> float:
        if not self.run_stats:
            return 0.0
        return statistics.mean(s.pct(label) for s in self.run_stats)

    def std_pct(self, label) -> float:
        if len(self.run_stats) < 2:
            return 0.0
        return statistics.stdev(s.pct(label) for s in self.run_stats)

    def avg_total(self) -> float:
        return statistics.mean(s.total for s in self.run_stats) if self.run_stats else 0

    def print_summary(self, show_examples: int = 1):
        print(f"Method : {self.name}")
        print(f"Runs   : {self.n_runs}  (avg {self.avg_total():.0f} completions/run)")
        print()
        for key in LABELS:
            avg = self.avg_pct(key)
            std = self.std_pct(key)
            per_run = '  '.join(f"run{i}:{s.pct(key):.1f}%" for i, s in enumerate(self.run_stats))
            print(f"  {LABEL_DISPLAY[key]:<28s} avg={avg:5.1f}%  std={std:4.1f}%   [{per_run}]")
        print()

        if show_examples:
            # Collect bad results across all runs
            by_label = {}
            for s in self.run_stats:
                for r in s.bad_results:
                    by_label.setdefault(r.label, []).append((s.path, r))
            for label, pairs in by_label.items():
                print(f"  ── {label} example ──")
                path, r = pairs[0]
                print(f"    {Path(path).parent.name}/{Path(path).name}  sample {r.index}: {r.detail}")
                print(f"    {r.snippet}")
                print()


def analyze_method(method_dir: str) -> MethodStats:
    """Discover run_*/completions.jsonl under method_dir and aggregate."""
    d = Path(method_dir)
    run_files = sorted(d.glob('run_*/completions.jsonl'))
    if not run_files:
        raise FileNotFoundError(f"No run_*/completions.jsonl found under {method_dir}")

    # Name: last two path components, e.g. "evict_cur / 512_per_head"
    parts = d.parts
    name = ' / '.join(parts[-2:]) if len(parts) >= 2 else d.name

    ms = MethodStats(name=name)
    for rf in run_files:
        ms.add(analyze_file(str(rf)))
    return ms


# ── comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(method_stats: list, mode: str = 'avg'):
    """Print a side-by-side table across methods.

    mode: 'avg'      → show avg ± std
          'per_run'  → show individual run percentages
    """
    COL = 22
    W = 16

    print('=' * (COL + W * len(method_stats) + 2))
    print('COMPARISON TABLE (averages across runs)')
    print('=' * (COL + W * len(method_stats) + 2))

    # Header row
    print(f"{'':>{COL}}", end='')
    for ms in method_stats:
        short = ms.name.split(' / ')[0]  # method name without config
        print(f"  {short[:W-2]:>{W-2}}", end='')
    print()
    print('-' * (COL + W * len(method_stats) + 2))

    for key in LABELS:
        print(f"  {LABEL_DISPLAY[key]:<{COL-2}}", end='')
        for ms in method_stats:
            avg = ms.avg_pct(key)
            std = ms.std_pct(key)
            cell = f"{avg:.1f}±{std:.1f}%"
            print(f"  {cell:>{W-2}}", end='')
        print()

    print()
    # Median completion length row
    print(f"  {'Median length':<{COL-2}}", end='')
    for ms in method_stats:
        all_lens = [l for s in ms.run_stats for l in s.lengths]
        med = statistics.median(all_lens) if all_lens else 0
        print(f"  {med:>{W-2}.0f}", end='')
    print()
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('files', nargs='*', default=[],
                       help='Individual completions.jsonl file(s)')
    group.add_argument('--methods', nargs='+', metavar='DIR',
                       help='Method directories containing run_*/completions.jsonl')
    parser.add_argument('--examples', type=int, default=1,
                        help='Number of examples to show per problem type (default: 1)')
    args = parser.parse_args()

    if args.methods:
        # ── method mode: average over runs ───────────────────────────────────
        all_method_stats = []
        for d in args.methods:
            try:
                ms = analyze_method(d)
            except FileNotFoundError as e:
                print(f"ERROR: {e}\n")
                continue
            all_method_stats.append(ms)
            print('=' * 70)
            ms.print_summary(show_examples=args.examples)

        if len(all_method_stats) > 1:
            print_comparison_table(all_method_stats)

    else:
        # ── file mode: analyze individual files ───────────────────────────────
        all_stats = []
        for path in args.files:
            if not Path(path).exists():
                print(f"ERROR: {path} not found\n")
                continue
            stats = analyze_file(path)
            all_stats.append(stats)
            print('=' * 70)
            stats.print_summary(show_examples=args.examples)

        if len(all_stats) > 1:
            # Simple file comparison table (no averaging)
            COL, W = 22, 18
            print('=' * (COL + W * len(all_stats) + 2))
            print('COMPARISON TABLE')
            print('=' * (COL + W * len(all_stats) + 2))
            print(f"{'':>{COL}}", end='')
            for s in all_stats:
                name = Path(s.path).parts[-4] if len(Path(s.path).parts) >= 4 else Path(s.path).stem
                print(f"  {name[:W-2]:>{W-2}}", end='')
            print()
            print('-' * (COL + W * len(all_stats) + 2))
            for key in LABELS:
                print(f"  {LABEL_DISPLAY[key]:<{COL-2}}", end='')
                for s in all_stats:
                    cell = f"{s.counts[key]} ({s.pct(key):.1f}%)"
                    print(f"  {cell:>{W-2}}", end='')
                print()
            print()


if __name__ == '__main__':
    main()
