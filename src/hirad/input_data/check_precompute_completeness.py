"""Check which IFS-HRES inits have all requested leads precomputed. Prints incomplete inits as
ISO base_dates for the ``base_dates.missing`` list of the anemoi build recipe (so a trajectory
build doesn't choke on a gap).

Example
-------
python -m hirad.input_data.check_precompute_completeness \
    --out-dir /capstor/scratch/cscs/pstamenk/ifs-hres-realch1/precompute_2020_202502 \
    --start 2020-10-01 --end 2025-02-28 --leads 1-33
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

from hirad.input_data.ifs_hres_precompute import out_name, parse_leads


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="precompute output dir")
    p.add_argument("--start", required=True, help="first init date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last init date YYYY-MM-DD")
    p.add_argument("--cycles", nargs="+", default=["00", "12"], help="init cycles (default 00 12)")
    p.add_argument("--leads", default="1-33", help="lead hours, e.g. '1-33'")
    args = p.parse_args()

    leads = parse_leads(args.leads)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    incomplete, complete, total = [], 0, 0
    d = start
    while d <= end:
        for hh in args.cycles:
            init = d.strftime("%y%m%d") + hh
            total += 1
            missing = [h for h in leads
                       if not os.path.exists(os.path.join(args.out_dir, out_name(init, h)))]
            if missing:
                iso = f"20{init[:2]}-{init[2:4]}-{init[4:6]}T{hh}:00:00"
                incomplete.append((iso, len(missing)))
            else:
                complete += 1
        d += dt.timedelta(days=1)

    print(f"inits: {total} total, {complete} complete, {len(incomplete)} incomplete")
    if incomplete:
        for iso, n in incomplete:
            print(f"  {iso}  (missing {n} leads)")
        print("\n# recipe base_dates.missing:")
        print("  missing: [" + ", ".join(f"'{iso}'" for iso, _ in incomplete) + "]")
    else:
        print("all inits complete -> base_dates.missing: []")


if __name__ == "__main__":
    main()
