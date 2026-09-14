"""
Render whatever result JSONs are present in results/ into a standalone SVG
table (real <text> elements, not outlined paths, so it stays
searchable/selectable) plus a markdown table printed to stdout for manual
use elsewhere.

    pixi run python benchmarks/bench_vs_upstream/scripts/render_table.py \
        --out docs/_static/benchmark_vs_upstream.svg
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("benchmarks/bench_vs_upstream/results")


def _load(results_dir):
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    if not rows:
        raise SystemExit(f"no *.json result files in {results_dir}; run run_all.sh first")
    # upstream (any thread count) first as the baseline, then fastcellstates configs
    rows.sort(key=lambda r: (r["tool"] != "cellstates (upstream)", r.get("threads", 0), r["config"]))
    return rows


def _fmt_time(s):
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"


def _fmt_singletons(r):
    n = r.get("n_singletons")
    return "?" if n is None else f"{n:,}"


COLUMNS = [
    ("", lambda r: f"{r['tool']} ({r['config']})"),
    ("states", lambda r: f"{r['n_states']:,}"),
    ("singlets", _fmt_singletons),
    (chr(0x398), lambda r: f"{r['theta']:,.0f}"),
    ("time", lambda r: _fmt_time(r["seconds"])),
    ("peak RAM", lambda r: f"{r['peak_rss_mb']:,.0f} MB"),
    ("log-likelihood", lambda r: f"{r['log_likelihood']:,.0f}"),
]


LL_COL = next(i for i, (name, _) in enumerate(COLUMNS) if name == "log-likelihood")


def render_svg(rows, out_path):
    row_h, pad = 34, 14
    col_w = [300, 90, 100, 90, 110, 170, 170]
    assert len(col_w) == len(COLUMNS)
    width = sum(col_w) + 2 * pad
    height = row_h * (len(rows) + 1) + 2 * pad

    def col_x(i):
        return pad + sum(col_w[:i])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Menlo, Consolas, monospace" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    # header
    hy = pad + row_h * 0.68
    for i, (name, _) in enumerate(COLUMNS):
        svg.append(f'<text x="{col_x(i) + 6}" y="{hy}" font-weight="700" fill="#111">{name}</text>')
    svg.append(
        f'<line x1="{pad}" y1="{pad + row_h}" x2="{width - pad}" y2="{pad + row_h}" '
        f'stroke="#333" stroke-width="1.5"/>'
    )

    best_ll_idx = max(range(len(rows)), key=lambda i: rows[i]["log_likelihood"])
    for r_i, row in enumerate(rows):
        y0 = pad + row_h * (r_i + 1)
        if r_i % 2 == 1:
            svg.append(f'<rect x="{pad}" y="{y0}" width="{width - 2 * pad}" height="{row_h}" fill="#f6f6f6"/>')
        ty = y0 + row_h * 0.68
        for c_i, (_, fmt) in enumerate(COLUMNS):
            text = fmt(row)
            weight = "700" if c_i == LL_COL and r_i == best_ll_idx else "400"
            color = "#0a7d2e" if c_i == LL_COL and r_i == best_ll_idx else "#111"
            svg.append(
                f'<text x="{col_x(c_i) + 6}" y="{ty}" font-weight="{weight}" fill="{color}">{text}</text>'
            )
    svg.append(
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="#ccc" stroke-width="1"/>'
    )
    svg.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))
    print(f"wrote {out_path}")


def print_markdown(rows):
    header = [name for name, _ in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        cells = [fmt(row) for _, fmt in COLUMNS]
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", default="docs/_static/benchmark_vs_upstream.svg")
    args = ap.parse_args()

    rows = _load(Path(args.results_dir))
    render_svg(rows, Path(args.out))
    print()
    print_markdown(rows)


if __name__ == "__main__":
    main()
