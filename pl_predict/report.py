"""Render the predicted table to a self-contained HTML page."""

import datetime

from pl_predict.config import PREDICT_SEASON_LABEL

_LABEL_TEXT = {
    "champion": "Champion",
    "top_4": "Top 4",
    "relegation": "Relegation",
    "mid_table": "",
}


def _rows_html(table):
    cells = []
    for _, r in table.iterrows():
        label = r["label"]
        badge = (
            f'<span class="badge {label}">{_LABEL_TEXT[label]}</span>'
            if _LABEL_TEXT[label]
            else ""
        )
        cells.append(
            f'<tr class="{label}">'
            f'<td class="rank">{r["predicted_rank"]}</td>'
            f'<td class="team">{r["team"]}</td>'
            f'<td class="pts">{r["predicted_points"]:.1f}</td>'
            f"<td>{badge}</td>"
            "</tr>"
        )
    return "\n".join(cells)


def write_html(table, metrics, path):
    """Write a standalone HTML report for the predicted season to `path`."""
    champion = table.loc[0, "team"]
    generated = datetime.date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Premier League {PREDICT_SEASON_LABEL} Prediction</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #0f1722; color: #e6edf3; }}
  .wrap {{ max-width: 720px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 .25rem; }}
  .sub {{ color: #9fb0c0; margin: 0 0 1.5rem; }}
  .champ {{ background: linear-gradient(135deg, #1f6feb33, #d2a8ff22);
            border: 1px solid #2d6cdf55; border-radius: 12px; padding: 1rem 1.25rem;
            margin-bottom: 1.5rem; }}
  .champ .lbl {{ color: #9fb0c0; font-size: .8rem; text-transform: uppercase;
                 letter-spacing: .08em; }}
  .champ .name {{ font-size: 1.5rem; font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
  th, td {{ text-align: left; padding: .55rem .6rem;
            border-bottom: 1px solid #ffffff14; }}
  th {{ color: #9fb0c0; font-weight: 600; font-size: .8rem;
        text-transform: uppercase; letter-spacing: .05em; }}
  td.rank {{ color: #9fb0c0; width: 2.5rem; }}
  td.team {{ font-weight: 600; }}
  td.pts {{ text-align: right; font-variant-numeric: tabular-nums; width: 5rem; }}
  tr.champion {{ background: #1f6feb1f; }}
  tr.top_4 {{ background: #1f6feb12; }}
  tr.relegation {{ background: #f8514912; }}
  .badge {{ font-size: .72rem; padding: .12rem .5rem; border-radius: 999px;
            font-weight: 600; }}
  .badge.champion {{ background: #d2a8ff33; color: #d2a8ff; }}
  .badge.top_4 {{ background: #1f6feb33; color: #79c0ff; }}
  .badge.relegation {{ background: #f8514933; color: #ff7b72; }}
  footer {{ margin-top: 1.75rem; color: #6e7d8c; font-size: .82rem;
            line-height: 1.5; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Premier League {PREDICT_SEASON_LABEL} — Predicted Table</h1>
  <p class="sub">Predicted from 2025-2026 results · generated {generated}</p>

  <div class="champ">
    <div class="lbl">Predicted champion</div>
    <div class="name">{champion}</div>
  </div>

  <table>
    <thead>
      <tr><th>#</th><th>Team</th><th style="text-align:right">Pts</th><th></th></tr>
    </thead>
    <tbody>
{_rows_html(table)}
    </tbody>
  </table>

  <footer>
    Model: StandardScaler + Ridge trained on season-to-season transitions
    (leave-one-out CV: MAE {metrics['mae']:.1f} pts, R&sup2; {metrics['r2']:.2f}).
    Predictions cover teams continuing from 2025-2026; promoted clubs are unknown,
    so the relegation zone shown is the three weakest continuing sides.
  </footer>
</div>
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(html)
