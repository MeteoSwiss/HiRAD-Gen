#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from eval.paths import DEFAULT_EVAL_ROOT


NEED_EVENTS = ["dora", "fernanda", "hilary", "idalia", "franklin"]


def _fmt_score(x: float) -> str:
    return "na" if not math.isfinite(x) else f"{x:.3f}"


def _fmt_frac(x: float) -> str:
    return "na" if not math.isfinite(x) else f"{x:.3e}"


def _tail_repro_score(exp_val: float | None, ref_val: float | None) -> float:
    if exp_val is None or ref_val is None or exp_val <= 0 or ref_val <= 0:
        return float("nan")
    ratio = exp_val / ref_val
    return math.exp(-abs(math.log(max(ratio, 1e-12))))


def build_prepml_rows(eval_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stats_files = sorted(glob.glob(str(eval_root / "*" / "tc_normed_pdfs_all_events_*.stats.json")))
    for stats_path in stats_files:
        with open(stats_path) as f:
            data = json.load(f)
        events = data.get("events", {})
        if not set(NEED_EVENTS).issubset(events.keys()):
            continue
        expver = data.get("expver") or Path(stats_path).stem.replace("tc_normed_pdfs_all_events_", "")
        exp = f"ENFO_O320_{expver}"

        event_scores: dict[str, float] = {}
        for ev in NEED_EVENTS:
            e = events[ev]
            m_curves = e.get("variables", {}).get("mslp_hpa", {}).get("curves", {})
            w_curves = e.get("variables", {}).get("wind10m_ms", {}).get("curves", {})
            ref_m = m_curves.get("ENFO_O320_ip6y", {}).get("tail", {})
            ref_w = w_curves.get("ENFO_O320_ip6y", {}).get("tail", {})
            exp_m = m_curves.get(exp, {}).get("tail", {})
            exp_w = w_curves.get(exp, {}).get("tail", {})
            m_score = _tail_repro_score(exp_m.get("bottom0.1_mean"), ref_m.get("bottom0.1_mean"))
            w_score = _tail_repro_score(exp_w.get("top0.1_mean"), ref_w.get("top0.1_mean"))
            if math.isfinite(m_score) and math.isfinite(w_score):
                event_scores[ev] = 0.5 * (m_score + w_score)
            else:
                event_scores[ev] = float("nan")

        valid_scores = [v for v in event_scores.values() if math.isfinite(v)]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")

        id_rows = events["idalia"].get("extreme_tail", {}).get("rows", [])
        id_self = next((r for r in id_rows if r.get("exp") == exp), {})

        rows.append(
            {
                "exp": exp,
                "expver": expver,
                "overall": overall,
                "dora": event_scores["dora"],
                "fernanda": event_scores["fernanda"],
                "hilary": event_scores["hilary"],
                "idalia": event_scores["idalia"],
                "franklin": event_scores["franklin"],
                "idalia_extreme_score": float(id_self.get("extreme_score", float("nan"))),
                "idalia_mslp_frac": float(id_self.get("mslp_980_990_fraction", float("nan"))),
                "idalia_wind_frac": float(id_self.get("wind_gt_25_fraction", float("nan"))),
                "source_file": stats_path,
            }
        )

    rows.sort(key=lambda r: (r["overall"] if math.isfinite(r["overall"]) else -1), reverse=True)
    return rows


def build_all_ml_rows(eval_root: Path) -> list[dict[str, object]]:
    by_exp: dict[str, dict[str, object]] = {}
    stats_files = sorted(glob.glob(str(eval_root / "*" / "tc_normed_pdfs_all_events_*.stats.json")))
    for stats_path in stats_files:
        path = Path(stats_path)
        mtime = path.stat().st_mtime
        with path.open() as f:
            data = json.load(f)
        events = data.get("events", {})
        if not set(NEED_EVENTS).issubset(events.keys()):
            continue

        # Candidate experiments from curve keys in this stats file
        curve_keys: set[str] = set()
        for ev in NEED_EVENTS:
            m_curves = events[ev].get("variables", {}).get("mslp_hpa", {}).get("curves", {})
            w_curves = events[ev].get("variables", {}).get("wind10m_ms", {}).get("curves", {})
            curve_keys |= set(m_curves.keys()) & set(w_curves.keys())

        for exp in sorted(curve_keys):
            event_scores: dict[str, float] = {}
            for ev in NEED_EVENTS:
                e = events[ev]
                m_curves = e.get("variables", {}).get("mslp_hpa", {}).get("curves", {})
                w_curves = e.get("variables", {}).get("wind10m_ms", {}).get("curves", {})
                ref_m = m_curves.get("ENFO_O320_ip6y", {}).get("tail", {})
                ref_w = w_curves.get("ENFO_O320_ip6y", {}).get("tail", {})
                exp_m = m_curves.get(exp, {}).get("tail", {})
                exp_w = w_curves.get(exp, {}).get("tail", {})
                m_score = _tail_repro_score(exp_m.get("bottom0.1_mean"), ref_m.get("bottom0.1_mean"))
                w_score = _tail_repro_score(exp_w.get("top0.1_mean"), ref_w.get("top0.1_mean"))
                if math.isfinite(m_score) and math.isfinite(w_score):
                    event_scores[ev] = 0.5 * (m_score + w_score)
                else:
                    event_scores[ev] = float("nan")

            valid_scores = [v for v in event_scores.values() if math.isfinite(v)]
            if not valid_scores:
                continue
            overall = sum(valid_scores) / len(valid_scores)
            id_rows = events["idalia"].get("extreme_tail", {}).get("rows", [])
            id_self = next((r for r in id_rows if r.get("exp") == exp), {})

            candidate = {
                "exp": exp,
                "overall": overall,
                "dora": event_scores["dora"],
                "fernanda": event_scores["fernanda"],
                "hilary": event_scores["hilary"],
                "idalia": event_scores["idalia"],
                "franklin": event_scores["franklin"],
                "idalia_extreme_score": float(id_self.get("extreme_score", float("nan"))),
                "idalia_mslp_frac": float(id_self.get("mslp_980_990_fraction", float("nan"))),
                "idalia_wind_frac": float(id_self.get("wind_gt_25_fraction", float("nan"))),
                "source_file": stats_path,
                "source_mtime": mtime,
            }

            prev = by_exp.get(exp)
            if prev is None or float(candidate["source_mtime"]) > float(prev["source_mtime"]):
                by_exp[exp] = candidate

    rows = list(by_exp.values())
    rows.sort(key=lambda r: (r["overall"] if math.isfinite(r["overall"]) else -1), reverse=True)
    return rows


def write_prepml_markdown(rows: list[dict[str, object]], out_file: Path) -> None:
    source_ids: OrderedDict[str, int] = OrderedDict()
    for r in rows:
        p = str(r["source_file"])
        if p not in source_ids:
            source_ids[p] = len(source_ids) + 1

    lines = []
    lines.append("# PrepML All-TC Reproduction Scoreboard")
    lines.append("")
    lines.append(
        "Event score compares each prepml exp to `ENFO_O320_ip6y` on extremes "
        "(MSLP `bottom0.1_mean`, Wind `top0.1_mean`) with `exp(-|ln(ratio)|)`."
    )
    lines.append("`1.0` means exact extreme-tail reproduction vs ip6y.")
    lines.append("")
    lines.append(f"Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append("")
    lines.append(
        "| Rank | Exp | Overall | Dora | Fernanda | Hilary | Idalia | Franklin | "
        "Idalia `extreme_score` | Idalia `mslp_980_990` | Idalia `wind_gt_25` | Src |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, start=1):
        src_id = source_ids[str(r["source_file"])]
        lines.append(
            f"| {i} | {r['exp']} | {_fmt_score(float(r['overall']))} | {_fmt_score(float(r['dora']))} | "
            f"{_fmt_score(float(r['fernanda']))} | {_fmt_score(float(r['hilary']))} | {_fmt_score(float(r['idalia']))} | "
            f"{_fmt_score(float(r['franklin']))} | {_fmt_score(float(r['idalia_extreme_score']))} | "
            f"{_fmt_frac(float(r['idalia_mslp_frac']))} | {_fmt_frac(float(r['idalia_wind_frac']))} | {src_id} |"
        )
    lines.append("")
    lines.append("## Sources")
    for p, sid in source_ids.items():
        lines.append(f"- `{sid}`: `{p}`")
    lines.append("")
    out_file.write_text("\n".join(lines))


def _write_html_table(
    *,
    out_file: Path,
    title: str,
    subtitle: str,
    generated_utc: str,
    columns: list[str],
    rows: list[list[str]],
    sources: OrderedDict[str, int] | None = None,
) -> None:
    col_html = "".join(f"<th>{escape(c)}</th>" for c in columns)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(v))}</td>" for v in row)
        row_html.append(f"<tr>{cells}</tr>")

    source_html = ""
    if sources:
        items = []
        for path, sid in sources.items():
            items.append(f"<li><code>{sid}</code>: <code>{escape(path)}</code></li>")
        source_html = (
            "<section><h2>Sources</h2>"
            f"<ul>{''.join(items)}</ul>"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f4ef;
      --ink: #1f2a37;
      --muted: #4a5565;
      --card: #ffffff;
      --head: #163a5f;
      --head-ink: #f8fafc;
      --line: #d5dde7;
      --zebra: #f9fbfd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top right, #d7e7f7, var(--bg) 45%);
      padding: 24px;
    }}
    main {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 1.7rem; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    .card {{
      margin-top: 16px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: auto;
      box-shadow: 0 8px 24px rgba(31, 42, 55, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.93rem;
      min-width: 980px;
    }}
    th, td {{ padding: 9px 12px; border-bottom: 1px solid var(--line); }}
    th {{
      position: sticky;
      top: 0;
      background: var(--head);
      color: var(--head-ink);
      text-align: left;
      letter-spacing: 0.01em;
      white-space: nowrap;
    }}
    tbody tr:nth-child(even) {{ background: var(--zebra); }}
    td:first-child, th:first-child {{ width: 64px; }}
    code {{
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 0.88em;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p>{escape(subtitle)}</p>
    <p>Generated (UTC): <code>{escape(generated_utc)}</code></p>
    <div class="card">
      <table>
        <thead><tr>{col_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    {source_html}
  </main>
</body>
</html>
"""
    out_file.write_text(html)


def write_prepml_html(rows: list[dict[str, object]], out_file: Path) -> None:
    source_ids: OrderedDict[str, int] = OrderedDict()
    for r in rows:
        p = str(r["source_file"])
        if p not in source_ids:
            source_ids[p] = len(source_ids) + 1

    html_rows: list[list[str]] = []
    for i, r in enumerate(rows, start=1):
        src_id = str(source_ids[str(r["source_file"])])
        html_rows.append(
            [
                str(i),
                str(r["exp"]),
                _fmt_score(float(r["overall"])),
                _fmt_score(float(r["dora"])),
                _fmt_score(float(r["fernanda"])),
                _fmt_score(float(r["hilary"])),
                _fmt_score(float(r["idalia"])),
                _fmt_score(float(r["franklin"])),
                _fmt_score(float(r["idalia_extreme_score"])),
                _fmt_frac(float(r["idalia_mslp_frac"])),
                _fmt_frac(float(r["idalia_wind_frac"])),
                src_id,
            ]
        )

    _write_html_table(
        out_file=out_file,
        title="PrepML All-TC Reproduction Scoreboard",
        subtitle="Extreme-tail reproduction vs ENFO_O320_ip6y across Dora/Fernanda/Hilary/Idalia/Franklin.",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        columns=[
            "Rank",
            "Exp",
            "Overall",
            "Dora",
            "Fernanda",
            "Hilary",
            "Idalia",
            "Franklin",
            "Idalia extreme_score",
            "Idalia mslp_980_990",
            "Idalia wind_gt_25",
            "Src",
        ],
        rows=html_rows,
        sources=source_ids,
    )


def write_all_ml_markdown(rows: list[dict[str, object]], out_file: Path) -> None:
    source_ids: OrderedDict[str, int] = OrderedDict()
    for r in rows:
        p = str(r["source_file"])
        if p not in source_ids:
            source_ids[p] = len(source_ids) + 1

    lines = []
    lines.append("# All-ML All-TC Reproduction Scoreboard")
    lines.append("")
    lines.append(
        "Includes all experiment curves available across all 5 events "
        "(`dora`, `fernanda`, `hilary`, `idalia`, `franklin`), using newest source per experiment."
    )
    lines.append("Reference for reproduction score is `ENFO_O320_ip6y`.")
    lines.append("")
    lines.append(f"Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append("")
    lines.append(
        "| Rank | Exp | Overall | Dora | Fernanda | Hilary | Idalia | Franklin | "
        "Idalia `extreme_score` | Idalia `mslp_980_990` | Idalia `wind_gt_25` | Src |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, start=1):
        src_id = source_ids[str(r["source_file"])]
        lines.append(
            f"| {i} | {r['exp']} | {_fmt_score(float(r['overall']))} | {_fmt_score(float(r['dora']))} | "
            f"{_fmt_score(float(r['fernanda']))} | {_fmt_score(float(r['hilary']))} | {_fmt_score(float(r['idalia']))} | "
            f"{_fmt_score(float(r['franklin']))} | {_fmt_score(float(r['idalia_extreme_score']))} | "
            f"{_fmt_frac(float(r['idalia_mslp_frac']))} | {_fmt_frac(float(r['idalia_wind_frac']))} | {src_id} |"
        )
    lines.append("")
    lines.append("## Sources")
    for p, sid in source_ids.items():
        lines.append(f"- `{sid}`: `{p}`")
    lines.append("")
    out_file.write_text("\n".join(lines))


def write_all_ml_html(rows: list[dict[str, object]], out_file: Path) -> None:
    source_ids: OrderedDict[str, int] = OrderedDict()
    for r in rows:
        p = str(r["source_file"])
        if p not in source_ids:
            source_ids[p] = len(source_ids) + 1

    html_rows: list[list[str]] = []
    for i, r in enumerate(rows, start=1):
        src_id = str(source_ids[str(r["source_file"])])
        html_rows.append(
            [
                str(i),
                str(r["exp"]),
                _fmt_score(float(r["overall"])),
                _fmt_score(float(r["dora"])),
                _fmt_score(float(r["fernanda"])),
                _fmt_score(float(r["hilary"])),
                _fmt_score(float(r["idalia"])),
                _fmt_score(float(r["franklin"])),
                _fmt_score(float(r["idalia_extreme_score"])),
                _fmt_frac(float(r["idalia_mslp_frac"])),
                _fmt_frac(float(r["idalia_wind_frac"])),
                src_id,
            ]
        )

    _write_html_table(
        out_file=out_file,
        title="All-ML All-TC Reproduction Scoreboard",
        subtitle="All experiment curves with complete event coverage; newest source file retained per experiment.",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        columns=[
            "Rank",
            "Exp",
            "Overall",
            "Dora",
            "Fernanda",
            "Hilary",
            "Idalia",
            "Franklin",
            "Idalia extreme_score",
            "Idalia mslp_980_990",
            "Idalia wind_gt_25",
            "Src",
        ],
        rows=html_rows,
        sources=source_ids,
    )


def write_global_markdown(tsv_file: Path, out_file: Path) -> None:
    rows: list[dict[str, str]] = []
    with tsv_file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    source_ids: OrderedDict[str, int] = OrderedDict()
    for r in rows:
        p = r["source_file"]
        if p not in source_ids:
            source_ids[p] = len(source_ids) + 1

    lines = []
    lines.append("# Global Extreme Scoreboard")
    lines.append("")
    lines.append(f"Generated from: `{tsv_file}`")
    lines.append(f"Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append("")
    lines.append(
        "| Rank | Exp | `extreme_repro_score` | `extreme_score` | "
        "`mslp_980_990_fraction` | `wind_gt_25_fraction` | Src |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        src_id = source_ids[r["source_file"]]
        lines.append(
            f"| {r['rank']} | {r['exp']} | {r['extreme_repro_score']} | {r['extreme_score']} | "
            f"{r['mslp_980_990_fraction']} | {r['wind_gt_25_fraction']} | {src_id} |"
        )
    lines.append("")
    lines.append("## Sources")
    for p, sid in source_ids.items():
        lines.append(f"- `{sid}`: `{p}`")
    lines.append("")
    out_file.write_text("\n".join(lines))


def write_global_html(tsv_file: Path, out_file: Path) -> None:
    rows: list[dict[str, str]] = []
    with tsv_file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    source_ids: OrderedDict[str, int] = OrderedDict()
    for r in rows:
        p = r["source_file"]
        if p not in source_ids:
            source_ids[p] = len(source_ids) + 1

    html_rows: list[list[str]] = []
    for r in rows:
        html_rows.append(
            [
                r["rank"],
                r["exp"],
                r["extreme_repro_score"],
                r["extreme_score"],
                r["mslp_980_990_fraction"],
                r["wind_gt_25_fraction"],
                str(source_ids[r["source_file"]]),
            ]
        )

    _write_html_table(
        out_file=out_file,
        title="Global Extreme Scoreboard",
        subtitle=f"Generated from {tsv_file}",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        columns=[
            "Rank",
            "Exp",
            "extreme_repro_score",
            "extreme_score",
            "mslp_980_990_fraction",
            "wind_gt_25_fraction",
            "Src",
        ],
        rows=html_rows,
        sources=source_ids,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clean markdown/html scoreboards for TC extremes.")
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    scoreboard_dir = eval_root / "scoreboards"
    scoreboard_dir.mkdir(parents=True, exist_ok=True)

    prep_rows = build_prepml_rows(eval_root)
    prep_md = scoreboard_dir / "prepml_all_tc_reproduction_scoreboard.md"
    prep_html = scoreboard_dir / "prepml_all_tc_reproduction_scoreboard.html"
    write_prepml_markdown(prep_rows, prep_md)
    write_prepml_html(prep_rows, prep_html)

    all_ml_rows = build_all_ml_rows(eval_root)
    all_ml_md = scoreboard_dir / "all_ml_all_tc_reproduction_scoreboard.md"
    all_ml_html = scoreboard_dir / "all_ml_all_tc_reproduction_scoreboard.html"
    write_all_ml_markdown(all_ml_rows, all_ml_md)
    write_all_ml_html(all_ml_rows, all_ml_html)

    global_tsv = eval_root / "tc_extreme_scoreboard_all_exps.tsv"
    global_md = scoreboard_dir / "global_extreme_scoreboard.md"
    global_html = scoreboard_dir / "global_extreme_scoreboard.html"
    write_global_markdown(global_tsv, global_md)
    write_global_html(global_tsv, global_html)

    print(prep_md)
    print(prep_html)
    print(all_ml_md)
    print(all_ml_html)
    print(global_md)
    print(global_html)


if __name__ == "__main__":
    main()
