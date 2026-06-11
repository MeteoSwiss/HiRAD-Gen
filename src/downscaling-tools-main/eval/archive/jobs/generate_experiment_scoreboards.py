#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.paths import DEFAULT_EVAL_ROOT, default_scoreboard_path, reference_spectra_dir


FIELDS = ["2t_sfc", "10u_sfc", "10v_sfc", "t_850", "z_500"]
FIELD_META = {
    "2t_sfc": ("2t", "sfc"),
    "10u_sfc": ("10u", "sfc"),
    "10v_sfc": ("10v", "sfc"),
    "t_850": ("t", "850"),
    "z_500": ("z", "500"),
}
AMPL_RE = re.compile(
    r"^ampl_(?P<date>\d{8})_(?P<step>\d+)_(?P<param>[a-z0-9]+)_(?P<level>[a-z0-9]+)_(?P<expid>[a-z0-9]+)_n(?P<number>\d+)\.npy$"
)


DEFAULT_BASELINE = [
    ("j24v", "j24v"),
    ("ip6y", "ip6y"),
    ("j2hh", "j2hh"),
    ("iz2q", "iz2q_no_training"),
    ("iz2p", "iz2p_50k_training"),
    ("j0ys", "j0ys_pretraining_1e6"),
    ("j1lh", "j1lh_new_lognormal_1e5"),
    ("j10b", "j10b"),
]
DEFAULT_SPECTRA_EXCLUDE = ["ip6y"]
DEFAULT_BASELINE_FILE = default_scoreboard_path("o96_o320_baseline_experiments.json")


@dataclass
class ExpInfo:
    expid: str
    label: str
    tc_keys: list[str]


def _parse_exp_list(value: str) -> list[ExpInfo]:
    out: list[ExpInfo] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            expid, label = item.split(":", 1)
            out.append(ExpInfo(expid=expid.strip(), label=label.strip(), tc_keys=[]))
        else:
            out.append(ExpInfo(expid=item, label=item, tc_keys=[]))
    return out


def _default_baseline_arg() -> str:
    return ",".join(f"{e}:{lbl}" for e, lbl in DEFAULT_BASELINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate default spectra + TC-extreme scoreboards for one experiment.")
    p.add_argument("--target-expid", required=True)
    p.add_argument("--target-label", default="")
    p.add_argument(
        "--baseline",
        default="",
        help="Optional comma-separated expid:label entries. If unset, read --baseline-file.",
    )
    p.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE)
    p.add_argument("--fields", default=",".join(FIELDS))
    p.add_argument("--samples-per-field", type=int, default=5)
    p.add_argument("--ai-spectra-root", default="/home/ecm5702/perm/ai_spectra")
    p.add_argument("--reference-root", default=str(reference_spectra_dir("enfo_o320")))
    p.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    p.add_argument("--output-dir", default="")
    p.add_argument(
        "--spectra-exclude",
        default=",".join(DEFAULT_SPECTRA_EXCLUDE),
        help="Comma-separated expids to exclude from spectra scoreboards.",
    )
    return p.parse_args()


def _load_baseline_file(path: Path) -> list[ExpInfo]:
    if not path.exists():
        return _parse_exp_list(_default_baseline_arg())
    with path.open() as f:
        data = json.load(f)
    exps = data.get("experiments", data if isinstance(data, list) else [])
    out: list[ExpInfo] = []
    for item in exps:
        if not isinstance(item, dict):
            continue
        expid = str(item.get("expid", "")).strip()
        if not expid:
            continue
        label = str(item.get("label", expid)).strip() or expid
        tc_keys = [str(x).strip() for x in item.get("tc_keys", []) if str(x).strip()]
        out.append(ExpInfo(expid=expid, label=label, tc_keys=tc_keys))
    return out


def _collect_tokens(ai_root: Path, ref_root: Path, expid: str, field: str) -> set[tuple[str, int, int]]:
    sdir = ai_root / expid / "spectra" / field
    if not sdir.exists():
        return set()
    param, level = FIELD_META[field]
    out: set[tuple[str, int, int]] = set()
    for fp in sorted(sdir.glob("ampl_*.npy")):
        m = AMPL_RE.match(fp.name)
        if not m:
            continue
        if m.group("expid") != expid or m.group("param") != param or m.group("level") != level:
            continue
        date = m.group("date")
        step = int(m.group("step"))
        number = int(m.group("number"))
        ref = ref_root / field / f"ampl_{date}_{step}_{param}_{level}_1_n{number}.npy"
        if ref.exists():
            out.add((date, step, number))
    return out


def _score_pair(exp_ampl: np.ndarray, ref_ampl: np.ndarray) -> float:
    n = min(len(exp_ampl), len(ref_ampl))
    if n < 4:
        return float("nan")
    ell = np.arange(n)
    e = exp_ampl[:n]
    r = ref_ampl[:n]
    keep = (ell >= 2) & np.isfinite(e) & np.isfinite(r) & (e > 0.0) & (r > 0.0)
    if not np.any(keep):
        return float("nan")
    ratio = np.maximum(e[keep], 1e-30) / np.maximum(r[keep], 1e-30)
    return float(np.mean(np.abs(np.log10(ratio))))


def _fmt(v: float) -> str:
    return "na" if not math.isfinite(v) else f"{v:.6f}"


def _fmt_frac(v: float) -> str:
    return "na" if not math.isfinite(v) else f"{v:.3e}"


def _format_fixed_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    sep = "-+-".join("-" * w for w in widths)
    head = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    lines = [head, sep]
    for r in rows:
        lines.append(" | ".join(r[i].ljust(widths[i]) for i in range(len(r))))
    return "\n".join(lines)


def _find_latest_extreme_rows(
    eval_root: Path, event_filter: str | None = None
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    by_exp: dict[str, dict[str, object]] = {}
    all_rows: list[dict[str, object]] = []
    files = sorted(
        glob.glob(str(eval_root / "*" / "tc_extreme_tail_*.json"))
        + glob.glob(str(eval_root / "*" / "tc_normed_pdfs_*from_predictions.stats.json"))
    )
    for fp in files:
        p = Path(fp)
        mtime = p.stat().st_mtime
        with p.open() as f:
            data = json.load(f)
        event = str(data.get("event", "")).strip().lower() if isinstance(data, dict) else ""
        if event_filter and event != event_filter.strip().lower():
            continue
        rows = []
        if isinstance(data, dict):
            if isinstance(data.get("rows"), list):
                rows = data["rows"]
            elif isinstance(data.get("extreme_tail", {}).get("rows"), list):
                rows = data["extreme_tail"]["rows"]
        for r in rows:
            exp = str(r.get("exp", "")).strip()
            if not exp:
                continue
            rec = {
                "exp": exp,
                "event": event or "unknown",
                "extreme_score": float(r.get("extreme_score", float("nan"))),
                "mslp_980_990_fraction": float(r.get("mslp_980_990_fraction", float("nan"))),
                "wind_gt_25_fraction": float(r.get("wind_gt_25_fraction", float("nan"))),
                "source_file": fp,
                "source_mtime": mtime,
            }
            all_rows.append(rec)
            prev = by_exp.get(exp)
            if prev is None or mtime > float(prev["source_mtime"]):
                by_exp[exp] = rec
    return by_exp, all_rows


def _build_alias_lookup(exps: list[ExpInfo]) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for e in exps:
        keys = [e.expid, e.label, f"ENFO_O320_{e.expid}", f"ENFO_O320_{e.label}", *e.tc_keys]
        for k in keys:
            kk = str(k).strip()
            if kk:
                lookup[kk] = (e.label, e.expid)
    return lookup


def _normalize_tc_rows(
    rows_by_exp: dict[str, dict[str, object]],
    alias_lookup: dict[str, tuple[str, str]],
    target_expid: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for exp_key, rec in rows_by_exp.items():
        alias, expid = alias_lookup.get(exp_key, (exp_key, exp_key))
        out.append(
            {
                "exp_key": exp_key,
                "expid": expid,
                "label": alias,
                "event": str(rec.get("event", "unknown")),
                "extreme_score": float(rec.get("extreme_score", float("nan"))),
                "mslp_980_990_fraction": float(rec.get("mslp_980_990_fraction", float("nan"))),
                "wind_gt_25_fraction": float(rec.get("wind_gt_25_fraction", float("nan"))),
                "source_file": str(rec.get("source_file", "missing")),
                "is_target": expid == target_expid or exp_key == target_expid or alias == target_expid,
            }
        )
    out = sorted(
        out,
        key=lambda r: (
            math.isfinite(float(r["extreme_score"])) is False,
            -float(r["extreme_score"]) if math.isfinite(float(r["extreme_score"])) else 1e9,
            str(r["label"]),
        ),
    )
    return out


def main() -> None:
    args = parse_args()
    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    if args.baseline.strip():
        baseline = _parse_exp_list(args.baseline)
    else:
        baseline = _load_baseline_file(Path(args.baseline_file))
    spectra_exclude = {x.strip() for x in args.spectra_exclude.split(",") if x.strip()}
    target_label = args.target_label.strip() or args.target_expid
    if not any(e.expid == args.target_expid for e in baseline):
        baseline.append(ExpInfo(expid=args.target_expid, label=target_label, tc_keys=[]))
    exps = baseline
    spectra_exps = [e for e in exps if e.expid not in spectra_exclude]

    ai_root = Path(args.ai_spectra_root)
    ref_root = Path(args.reference_root)
    eval_root = Path(args.eval_root)
    out_dir = Path(args.output_dir) if args.output_dir else (eval_root / args.target_expid / "scoreboards")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spectra scoreboard
    sample_manifest: dict[str, list[dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for field in fields:
        token_sets_all = {e.expid: _collect_tokens(ai_root, ref_root, e.expid, field) for e in spectra_exps}
        providers = [s for s in token_sets_all.values() if s]
        common = set.intersection(*providers) if providers else set()
        chosen = sorted(common)[: args.samples_per_field]
        sample_manifest[field] = [{"date": d, "step": s, "number": n} for (d, s, n) in chosen]
        param, level = FIELD_META[field]
        for e in spectra_exps:
            vals = []
            for d, s, n in chosen:
                efp = ai_root / e.expid / "spectra" / field / f"ampl_{d}_{s}_{param}_{level}_{e.expid}_n{n}.npy"
                rfp = ref_root / field / f"ampl_{d}_{s}_{param}_{level}_1_n{n}.npy"
                if not efp.exists() or not rfp.exists():
                    continue
                vals.append(_score_pair(np.load(efp), np.load(rfp)))
            rows.append(
                {
                    "field": field,
                    "expid": e.expid,
                    "label": e.label,
                    "samples": len(chosen),
                    "score": float(np.mean(vals)) if vals else float("nan"),
                    "is_target": e.expid == args.target_expid,
                }
            )

    # Aggregate overall spectra score
    overall: list[dict[str, object]] = []
    for e in spectra_exps:
        vals = [r["score"] for r in rows if r["expid"] == e.expid and math.isfinite(float(r["score"]))]
        overall.append(
            {
                "expid": e.expid,
                "label": e.label,
                "overall_score": float(np.mean(vals)) if vals else float("nan"),
                "n_fields": len(vals),
                "is_target": e.expid == args.target_expid,
            }
        )
    overall = sorted(overall, key=lambda r: (math.isfinite(float(r["overall_score"])) is False, float(r["overall_score"]) if math.isfinite(float(r["overall_score"])) else 1e9))

    spectra_csv = out_dir / "spectra_scoreboard.csv"
    with spectra_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "expid", "label", "overall_score", "n_fields", "is_target"])
        w.writeheader()
        for i, r in enumerate(overall, 1):
            w.writerow({"rank": i, **r})

    # TC extremes scoreboards:
    # - global: all available events/experiments with extreme stats
    # - idalia: idalia-only rows + ensure manual_* experiments present when available
    alias_lookup = _build_alias_lookup(exps)

    extreme_map_all, _ = _find_latest_extreme_rows(eval_root, event_filter=None)
    tc_rows_all = _normalize_tc_rows(extreme_map_all, alias_lookup, args.target_expid)

    extreme_map_idalia, extreme_rows_idalia = _find_latest_extreme_rows(eval_root, event_filter="idalia")
    manual_idalia_files = sorted(glob.glob(str(eval_root / "manual_*" / "tc_extreme_tail_idalia*.json")))
    for fp in manual_idalia_files:
        manual_name = Path(fp).parent.name
        if manual_name in extreme_map_idalia:
            continue
        # Source-path fallback: pick the newest idalia record produced in this manual directory.
        path_hits = [r for r in extreme_rows_idalia if f"/{manual_name}/" in str(r.get("source_file", ""))]
        if not path_hits:
            continue
        extreme_map_idalia[manual_name] = sorted(
            path_hits, key=lambda r: float(r.get("source_mtime", 0.0)), reverse=True
        )[0]

    tc_rows_idalia = _normalize_tc_rows(extreme_map_idalia, alias_lookup, args.target_expid)

    tc_csv = out_dir / "tc_extreme_scoreboard.csv"
    with tc_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "event",
                "exp_key",
                "expid",
                "label",
                "extreme_score",
                "mslp_980_990_fraction",
                "wind_gt_25_fraction",
                "source_file",
                "is_target",
            ],
        )
        w.writeheader()
        for i, r in enumerate(tc_rows_all, 1):
            w.writerow({"rank": i, **r})

    tc_idalia_csv = out_dir / "tc_idalia_scoreboard.csv"
    with tc_idalia_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "event",
                "exp_key",
                "expid",
                "label",
                "extreme_score",
                "mslp_980_990_fraction",
                "wind_gt_25_fraction",
                "source_file",
                "is_target",
            ],
        )
        w.writeheader()
        for i, r in enumerate(tc_rows_idalia, 1):
            w.writerow({"rank": i, **r})

    # Markdown report with both scoreboards
    md = []
    md.append(f"# Experiment Scoreboards: {args.target_expid}")
    md.append("")
    md.append(f"Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    md.append(f"Target experiment: `{args.target_expid}` (rows marked with `*`)")
    md.append("")
    md.append(f"Baseline file: `{args.baseline_file}`")
    md.append("")
    md.append(f"Spectra excludes: `{','.join(sorted(spectra_exclude)) if spectra_exclude else '(none)'}`")
    md.append("")
    md.append("## Spectra Scoreboard (nice amplitudes, 5-sample policy)")
    md.append("")
    md.append("| Rank | * | Expid | Label | Overall score | Fields with score |")
    md.append("|---:|---|---|---|---:|---:|")
    for i, r in enumerate(overall, 1):
        star = "*" if bool(r["is_target"]) else ""
        md.append(
            f"| {i} | {star} | {r['expid']} | {r['label']} | {_fmt(float(r['overall_score']))} | {int(r['n_fields'])} |"
        )
    md.append("")
    for field in fields:
        md.append(f"### {field}")
        md.append("")
        md.append("| Rank | * | Expid | Label | Score | Samples |")
        md.append("|---:|---|---|---|---:|---:|")
        frows = [r for r in rows if r["field"] == field]
        frows = sorted(
            frows,
            key=lambda r: (
                math.isfinite(float(r["score"])) is False,
                float(r["score"]) if math.isfinite(float(r["score"])) else 1e9,
            ),
        )
        for j, r in enumerate(frows, 1):
            star = "*" if bool(r["is_target"]) else ""
            md.append(
                f"| {j} | {star} | {r['expid']} | {r['label']} | {_fmt(float(r['score']))} | {int(r['samples'])} |"
            )
        md.append("")

    md.append("## TC Extreme Scoreboard (all available)")
    md.append("")
    md.append("| Rank | * | Event | Label | Expid | ExpKey | extreme_score | mslp_980_990_fraction | wind_gt_25_fraction |")
    md.append("|---:|---|---|---|---|---|---:|---:|---:|")
    for i, r in enumerate(tc_rows_all, 1):
        star = "*" if bool(r["is_target"]) else ""
        md.append(
            f"| {i} | {star} | {r['event']} | {r['label']} | {r['expid']} | {r['exp_key']} | {_fmt(float(r['extreme_score']))} | {_fmt_frac(float(r['mslp_980_990_fraction']))} | {_fmt_frac(float(r['wind_gt_25_fraction']))} |"
        )
    md.append("")
    md.append("## TC Idalia Scoreboard (includes manual_* when available)")
    md.append("")
    md.append("| Rank | * | Label | Expid | ExpKey | extreme_score | mslp_980_990_fraction | wind_gt_25_fraction |")
    md.append("|---:|---|---|---|---|---:|---:|---:|")
    for i, r in enumerate(tc_rows_idalia, 1):
        star = "*" if bool(r["is_target"]) else ""
        md.append(
            f"| {i} | {star} | {r['label']} | {r['expid']} | {r['exp_key']} | {_fmt(float(r['extreme_score']))} | {_fmt_frac(float(r['mslp_980_990_fraction']))} | {_fmt_frac(float(r['wind_gt_25_fraction']))} |"
        )
    md.append("")
    md.append("## Notes")
    md.append("- `na` means missing required input data/artifacts.")
    md.append("- Spectra scores use mean `|log10(exp/reference)|` over selected samples; lower is better.")
    md.append("")
    md.append("## Artifacts")
    md.append(f"- `spectra_scoreboard.csv`: `{spectra_csv}`")
    md.append(f"- `tc_extreme_scoreboard.csv`: `{tc_csv}`")
    md.append(f"- `tc_idalia_scoreboard.csv`: `{tc_idalia_csv}`")
    md.append(f"- `sample_manifest.json`: `{out_dir / 'sample_manifest.json'}`")
    report_path = out_dir / "SCOREBOARDS.md"
    report_path.write_text("\n".join(md) + "\n")
    (out_dir / "sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2))

    # Fixed-width text report for terminal readability
    pretty_lines: list[str] = []
    pretty_lines.append(f"Experiment Scoreboards: {args.target_expid}")
    pretty_lines.append(f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    pretty_lines.append(f"Spectra excludes: {','.join(sorted(spectra_exclude)) if spectra_exclude else '(none)'}")
    pretty_lines.append("")
    pretty_lines.append("Spectra Scoreboard (overall)")
    overall_rows = []
    for i, r in enumerate(overall, 1):
        overall_rows.append(
            [
                str(i),
                "*" if bool(r["is_target"]) else "",
                str(r["label"]),
                str(r["expid"]),
                _fmt(float(r["overall_score"])),
                str(int(r["n_fields"])),
            ]
        )
    pretty_lines.append(
        _format_fixed_table(
            ["Rank", "*", "Label", "Expid", "OverallScore", "Fields"],
            overall_rows,
        )
    )
    pretty_lines.append("")
    for field in fields:
        pretty_lines.append(f"{field} ranking")
        frows = [r for r in rows if r["field"] == field]
        frows = sorted(
            frows,
            key=lambda r: (
                math.isfinite(float(r["score"])) is False,
                float(r["score"]) if math.isfinite(float(r["score"])) else 1e9,
            ),
        )
        trows: list[list[str]] = []
        for j, r in enumerate(frows, 1):
            trows.append(
                [
                    str(j),
                    "*" if bool(r["is_target"]) else "",
                    str(r["label"]),
                    str(r["expid"]),
                    _fmt(float(r["score"])),
                    str(int(r["samples"])),
                ]
            )
        pretty_lines.append(_format_fixed_table(["Rank", "*", "Label", "Expid", "Score", "Samples"], trows))
        pretty_lines.append("")
    pretty_lines.append("TC Extreme Scoreboard (all available)")
    tc_trows = []
    for i, r in enumerate(tc_rows_all, 1):
        tc_trows.append(
            [
                str(i),
                "*" if bool(r["is_target"]) else "",
                str(r["event"]),
                str(r["label"]),
                str(r["expid"]),
                str(r["exp_key"]),
                _fmt(float(r["extreme_score"])),
                _fmt_frac(float(r["mslp_980_990_fraction"])),
                _fmt_frac(float(r["wind_gt_25_fraction"])),
            ]
        )
    pretty_lines.append(
        _format_fixed_table(
            ["Rank", "*", "Event", "Label", "Expid", "ExpKey", "ExtremeScore", "MSLP_980_990", "Wind_gt_25"],
            tc_trows,
        )
    )
    pretty_lines.append("")
    pretty_lines.append("TC Idalia Scoreboard (includes manual_*)")
    tc_idalia_trows = []
    for i, r in enumerate(tc_rows_idalia, 1):
        tc_idalia_trows.append(
            [
                str(i),
                "*" if bool(r["is_target"]) else "",
                str(r["label"]),
                str(r["expid"]),
                str(r["exp_key"]),
                _fmt(float(r["extreme_score"])),
                _fmt_frac(float(r["mslp_980_990_fraction"])),
                _fmt_frac(float(r["wind_gt_25_fraction"])),
            ]
        )
    pretty_lines.append(
        _format_fixed_table(
            ["Rank", "*", "Label", "Expid", "ExpKey", "ExtremeScore", "MSLP_980_990", "Wind_gt_25"],
            tc_idalia_trows,
        )
    )
    pretty_path = out_dir / "SCOREBOARDS_PRETTY.txt"
    pretty_path.write_text("\n".join(pretty_lines) + "\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote spectra CSV: {spectra_csv}")
    print(f"Wrote TC (all-events) CSV: {tc_csv}")
    print(f"Wrote TC (idalia) CSV: {tc_idalia_csv}")
    print(f"Wrote sample manifest: {out_dir / 'sample_manifest.json'}")
    print(f"Wrote fixed-width report: {pretty_path}")


if __name__ == "__main__":
    main()
