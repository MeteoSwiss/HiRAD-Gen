#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${EVAL_ROOT:-/home/ecm5702/scratch/eval}"
OUT_FILE="${BASE_DIR}/tc_extreme_scoreboard_all_exps.tsv"

tmp="$(mktemp)"

find "${BASE_DIR}" -type f \( -name 'tc_extreme_tail_*.json' -o -name 'tc_normed_pdfs_*from_predictions.stats.json' \) | sort | while read -r f; do
  mtime="$(stat -c %Y "$f")"
  if jq -e '.rows' "$f" >/dev/null 2>&1; then
    jq -r --arg f "$f" --arg m "$mtime" \
      '.rows[] | select(has("exp")) | [$f,$m,.exp,((.extreme_repro_score // "na")|tostring),((.extreme_score // "na")|tostring),((.mslp_980_990_fraction // "na")|tostring),((.wind_gt_25_fraction // "na")|tostring)] | @tsv' \
      "$f"
  else
    jq -r --arg f "$f" --arg m "$mtime" \
      '.extreme_tail.rows[]? | [$f,$m,.exp,((.extreme_repro_score // "na")|tostring),((.extreme_score // "na")|tostring),((.mslp_980_990_fraction // "na")|tostring),((.wind_gt_25_fraction // "na")|tostring)] | @tsv' \
      "$f"
  fi
done | awk -F'\t' 'BEGIN{OFS="\t"} {if(!($3 in seen) || $2>seen[$3]){seen[$3]=$2; row[$3]=$0}} END{for(k in row) print row[k]}' \
  | awk -F'\t' 'BEGIN{OFS="\t"} {k=($4!="na"?$4:$5); print k,$0}' \
  | sort -t $'\t' -k1,1gr \
  | awk -F'\t' 'BEGIN{OFS="\t"; print "rank","exp","extreme_repro_score","extreme_score","mslp_980_990_fraction","wind_gt_25_fraction","source_file","source_mtime_unix"} {print NR,$4,$5,$6,$7,$8,$2,$3}' \
  > "$tmp"

mv "$tmp" "$OUT_FILE"
echo "Wrote scoreboard: $OUT_FILE"
