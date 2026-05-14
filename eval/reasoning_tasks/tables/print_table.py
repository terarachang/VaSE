import json
import sys

def print_table(path):
    with open(path) as f:
        data = json.load(f)

    benchmarks = list(data.keys())
    methods = [entry["Method"] for entry in next(iter(data.values()))]

    col_width = 20
    header = f"{'Method':<{col_width}}" + "".join(f"{b:>{col_width}}" for b in benchmarks) + f"{'Avg':>{col_width}}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<{col_width}}"
        vals = []
        for bench in benchmarks:
            entry = next((e for e in data[bench] if e["Method"] == method), None)
            val = entry["Acc"] if entry and entry["Acc"] is not None else None
            cell = f"{val:.2f}" if isinstance(val, float) else ""
            if val is not None and bench != "LiveCodeBench":
                vals.append(val)
            row += f"{cell:>{col_width}}"
        avg = f"{sum(vals)/len(vals):.2f}" if vals else ""
        row += f"{avg:>{col_width}}"
        print(row)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tables/all_Qwen3-14B.json"
    print_table(path)
