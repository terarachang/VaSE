import json
import sys

def print_budget_table(path):
    with open(path) as f:
        data = json.load(f)

    budgets = list(data.keys())
    methods = [entry["Method"] for entry in next(iter(data.values()))]

    col_width = 20
    header = f"{'Method':<{col_width}}" + "".join(f"{b:>{col_width}}" for b in budgets)
    print(header)
    print("-" * len(header))

    for method in methods:
        if method == "Full":
            print("-" * len(header))
        row = f"{method:<{col_width}}"
        for budget in budgets:
            entry = next((e for e in data[budget] if e["Method"] == method), None)
            val = entry["Acc"] if entry and entry["Acc"] is not None else None
            cell = f"{val:.2f}" if isinstance(val, float) else ""
            row += f"{cell:>{col_width}}"
        print(row)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tables/hmmt25_Qwen3-14B_budget.json"
    print_budget_table(path)
