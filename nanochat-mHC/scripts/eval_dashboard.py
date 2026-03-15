"""
generate a self-contained static HTML dashboard from eval results JSON.

usage:
    # from a specific results file
    python -m scripts.eval_dashboard path/to/results.json

    # from latest results
    python -m scripts.eval_dashboard --latest

    # custom output path
    python -m scripts.eval_dashboard results.json -o comparison.html
"""

import os
import json
import glob
import argparse
import webbrowser
from datetime import datetime

from nanochat.common import get_base_dir


def find_latest_results():
    """find the most recent results JSON file."""
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, "eval_results")
    files = glob.glob(os.path.join(results_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"no results found in {results_dir}")
    return max(files, key=os.path.getmtime)


def pct(value, decimals=2):
    """format a float as percentage string."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def diff_cell(a, b):
    """html for a diff cell comparing two values. green if b > a, red if b < a."""
    if a is None or b is None:
        return '<td class="diff">-</td>'
    delta = b - a
    pct_delta = delta * 100
    if abs(delta) < 1e-6:
        color = "#888"
        sign = ""
    elif delta > 0:
        color = "#2e7d32"
        sign = "+"
    else:
        color = "#c62828"
        sign = ""
    return f'<td class="diff" style="color:{color};font-weight:600">{sign}{pct_delta:.2f}pp</td>'


def bar_html(values, labels, colors, max_val=None, title=""):
    """generate css-only horizontal bar chart."""
    if max_val is None:
        max_val = max((v for v in values if v is not None), default=1)
    if max_val == 0:
        max_val = 1

    rows = []
    for val, label, color in zip(values, labels, colors):
        if val is None:
            width = 0
            display = "n/a"
        else:
            width = (val / max_val) * 100
            display = f"{val * 100:.1f}%"
        rows.append(f"""
        <div class="bar-row">
            <span class="bar-label">{label}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>
            </div>
            <span class="bar-value">{display}</span>
        </div>""")

    return f"""
    <div class="bar-chart">
        <div class="bar-title">{title}</div>
        {"".join(rows)}
    </div>"""


MODEL_COLORS = ["#1565c0", "#e65100", "#2e7d32", "#6a1b9a", "#c62828", "#00838f"]


def generate_dashboard(input_path, output_path=None, open_browser=True):
    """generate HTML dashboard from results JSON."""
    with open(input_path) as f:
        data = json.load(f)

    if output_path is None:
        output_path = input_path.replace(".json", ".html")

    timestamp = data.get("timestamp", "unknown")
    gpu_info = data.get("gpu_info", {})
    eval_config = data.get("eval_config", {})
    models = data.get("models", [])

    gpu_str = gpu_info.get("gpu_name", "unknown")
    if gpu_info.get("gpu_vram_gb"):
        gpu_str += f" ({gpu_info['gpu_vram_gb']}GB)"

    num_models = len(models)

    # build summary cards
    summary_cards = []
    for i, m in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        core_str = ""
        chat_str = ""
        time_str = ""

        total_time = 0
        if "core" in m:
            core_str = f'<div class="card-metric">CORE: {m["core"]["core_metric"]:.4f}</div>'
            total_time += m["core"]["wall_time"]
        if "chat" in m:
            cm = m["chat"].get("chatcore_metric")
            cm_str = f"{cm:.4f}" if cm is not None else "n/a"
            chat_str = f'<div class="card-metric">ChatCORE: {cm_str}</div>'
            total_time += m["chat"]["wall_time"]

        time_str = f'<div class="card-sub">eval time: {total_time:.1f}s</div>'

        summary_cards.append(f"""
        <div class="card" style="border-top:4px solid {color}">
            <div class="card-name">{m["model_name"]}</div>
            <div class="card-sub">{m["param_count"]:,} params</div>
            {core_str}
            {chat_str}
            {time_str}
        </div>""")

    # build CORE benchmark table
    core_table = ""
    if any("core" in m for m in models):
        # collect all task names from first model that has core results
        core_model = next(m for m in models if "core" in m)
        tasks = list(core_model["core"]["results"].keys())

        # header
        header_cells = "<th>Task</th>"
        for i, m in enumerate(models):
            if "core" not in m:
                continue
            header_cells += f'<th>{m["model_name"]}<br><small>accuracy</small></th>'
            header_cells += f'<th>{m["model_name"]}<br><small>centered</small></th>'
        if num_models >= 2 and all("core" in m for m in models):
            header_cells += "<th>diff (centered)</th>"

        # rows
        rows = []
        for task in tasks:
            cells = f"<td>{task}</td>"
            centered_vals = []
            for m in models:
                if "core" not in m:
                    continue
                acc = m["core"]["results"].get(task)
                cent = m["core"]["centered_results"].get(task)
                centered_vals.append(cent)
                cells += f"<td>{pct(acc)}</td>"
                cells += f"<td>{pct(cent)}</td>"
            if num_models >= 2 and len(centered_vals) >= 2:
                cells += diff_cell(centered_vals[0], centered_vals[-1])
            rows.append(f"<tr>{cells}</tr>")

        # summary row
        summary_cells = "<td><strong>CORE metric</strong></td>"
        core_metrics = []
        for m in models:
            if "core" not in m:
                continue
            cm = m["core"]["core_metric"]
            core_metrics.append(cm)
            summary_cells += f'<td></td><td><strong>{pct(cm)}</strong></td>'
        if num_models >= 2 and len(core_metrics) >= 2:
            summary_cells += diff_cell(core_metrics[0], core_metrics[-1])
        rows.append(f'<tr class="summary-row">{summary_cells}</tr>')

        core_table = f"""
        <h2>CORE Benchmark</h2>
        <table>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>"""

    # build ChatCORE benchmark table
    chat_table = ""
    if any("chat" in m for m in models):
        header_cells = "<th>Task</th>"
        for m in models:
            if "chat" not in m:
                continue
            header_cells += f'<th>{m["model_name"]}</th>'
        if num_models >= 2 and all("chat" in m for m in models):
            header_cells += "<th>diff</th>"

        rows = []
        for task in CHAT_TASKS:
            cells = f"<td>{task}</td>"
            vals = []
            for m in models:
                if "chat" not in m:
                    continue
                acc = m["chat"]["results"].get(task)
                vals.append(acc)
                cells += f"<td>{pct(acc)}</td>"
            if num_models >= 2 and len(vals) >= 2:
                cells += diff_cell(vals[0], vals[-1])
            rows.append(f"<tr>{cells}</tr>")

        # summary row
        summary_cells = "<td><strong>ChatCORE metric</strong></td>"
        metrics = []
        for m in models:
            if "chat" not in m:
                continue
            cm = m["chat"].get("chatcore_metric")
            metrics.append(cm)
            summary_cells += f'<td><strong>{pct(cm)}</strong></td>'
        if num_models >= 2 and len(metrics) >= 2:
            summary_cells += diff_cell(metrics[0], metrics[-1])
        rows.append(f'<tr class="summary-row">{summary_cells}</tr>')

        chat_table = f"""
        <h2>ChatCORE Benchmark</h2>
        <table>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>"""

    # build throughput table
    throughput_rows = []
    for m in models:
        name = m["model_name"]
        params = f'{m["param_count"]:,}'
        core_time = f'{m["core"]["wall_time"]:.1f}s' if "core" in m else "-"
        chat_time = f'{m["chat"]["wall_time"]:.1f}s' if "chat" in m else "-"
        total_time = 0
        if "core" in m:
            total_time += m["core"]["wall_time"]
        if "chat" in m:
            total_time += m["chat"]["wall_time"]
        throughput_rows.append(f"""
        <tr>
            <td>{name}</td>
            <td>{params}</td>
            <td>{core_time}</td>
            <td>{chat_time}</td>
            <td><strong>{total_time:.1f}s</strong></td>
        </tr>""")

    throughput_table = f"""
    <h2>Throughput / Timing</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Params</th>
                <th>CORE time</th>
                <th>ChatCORE time</th>
                <th>Total time</th>
            </tr>
        </thead>
        <tbody>{"".join(throughput_rows)}</tbody>
    </table>"""

    # build bar charts for CORE tasks
    bar_charts = ""
    if any("core" in m for m in models) and num_models >= 1:
        core_model = next(m for m in models if "core" in m)
        tasks = list(core_model["core"]["centered_results"].keys())

        chart_sections = []
        for task in tasks:
            values = []
            labels = []
            colors = []
            for i, m in enumerate(models):
                if "core" not in m:
                    continue
                val = m["core"]["centered_results"].get(task)
                values.append(val)
                labels.append(m["model_name"])
                colors.append(MODEL_COLORS[i % len(MODEL_COLORS)])
            chart_sections.append(bar_html(values, labels, colors, max_val=1.0, title=task))

        bar_charts = f"""
        <h2>CORE Task Comparison</h2>
        <div class="charts-grid">
            {"".join(chart_sections)}
        </div>"""

    # config info
    config_str = ""
    if eval_config.get("max_per_task", -1) != -1:
        config_str += f" | max_per_task: {eval_config['max_per_task']}"
    if eval_config.get("max_problems") is not None:
        config_str += f" | max_problems: {eval_config['max_problems']}"

    # model config details
    model_configs = ""
    for m in models:
        cfg = m.get("config", {})
        if cfg:
            cfg_items = [f"{k}: {v}" for k, v in cfg.items() if v is not None]
            model_configs += f'<div class="config-line"><strong>{m["model_name"]}</strong>: {", ".join(cfg_items)}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nanochat eval comparison - {timestamp}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #f5f5f5;
        color: #333;
        padding: 24px;
        max-width: 1200px;
        margin: 0 auto;
    }}
    h1 {{
        font-size: 1.5rem;
        margin-bottom: 4px;
    }}
    .header-sub {{
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 20px;
    }}
    h2 {{
        font-size: 1.15rem;
        margin: 28px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #ddd;
    }}
    .cards {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }}
    .card {{
        background: #fff;
        border-radius: 8px;
        padding: 16px 20px;
        min-width: 200px;
        flex: 1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .card-name {{
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 4px;
    }}
    .card-sub {{
        color: #666;
        font-size: 0.8rem;
        margin-bottom: 6px;
    }}
    .card-metric {{
        font-size: 1.1rem;
        font-weight: 600;
        margin: 4px 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: #fff;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 0.85rem;
    }}
    th {{
        background: #f0f0f0;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #ddd;
    }}
    td {{
        padding: 8px 12px;
        border-bottom: 1px solid #eee;
    }}
    tr:hover {{ background: #fafafa; }}
    .summary-row {{
        background: #f8f8f8;
        border-top: 2px solid #ddd;
    }}
    .summary-row td {{ font-weight: 600; }}
    .diff {{ text-align: center; }}
    .config-line {{
        font-size: 0.8rem;
        color: #666;
        margin: 2px 0;
    }}
    .charts-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 16px;
    }}
    .bar-chart {{
        background: #fff;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .bar-title {{
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }}
    .bar-row {{
        display: flex;
        align-items: center;
        margin: 4px 0;
        font-size: 0.8rem;
    }}
    .bar-label {{
        width: 140px;
        flex-shrink: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .bar-track {{
        flex: 1;
        height: 18px;
        background: #f0f0f0;
        border-radius: 3px;
        margin: 0 8px;
        overflow: hidden;
    }}
    .bar-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s;
    }}
    .bar-value {{
        width: 50px;
        text-align: right;
        flex-shrink: 0;
        font-weight: 600;
    }}
    .footer {{
        margin-top: 32px;
        text-align: center;
        color: #999;
        font-size: 0.75rem;
    }}
</style>
</head>
<body>
    <h1>nanochat eval comparison</h1>
    <div class="header-sub">
        {timestamp} | {gpu_str} | evals: {", ".join(eval_config.get("evals", []))}{config_str}
    </div>

    <div class="cards">
        {"".join(summary_cards)}
    </div>

    {model_configs}

    {core_table}
    {chat_table}
    {throughput_table}
    {bar_charts}

    <div class="footer">generated by nanochat eval_dashboard.py</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    if open_browser:
        try:
            webbrowser.open(f"file://{os.path.abspath(output_path)}")
        except Exception:
            pass

    return output_path


CHAT_TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]


def main():
    parser = argparse.ArgumentParser(description="generate eval dashboard from results JSON")
    parser.add_argument("input", nargs="?", default=None, help="path to results JSON file")
    parser.add_argument("--latest", action="store_true", help="use latest results file")
    parser.add_argument("-o", "--output", type=str, default=None, help="output HTML path")
    parser.add_argument("--no-open", action="store_true", help="don't open in browser")
    args = parser.parse_args()

    if args.latest or args.input is None:
        input_path = find_latest_results()
        print(f"using latest results: {input_path}")
    else:
        input_path = args.input

    output_path = generate_dashboard(input_path, args.output, open_browser=not args.no_open)
    print(f"dashboard written to {output_path}")


if __name__ == "__main__":
    main()
