# Phase8 Daily/Intraday Integration Design

## Goal

Add Phase8 short-term limit-up opportunity scores to daily and intraday screening without changing the existing main watchlist ranking.

## Decisions

- Phase8 is display-only for now.
- Daily screening runs Phase8 after Phase4 and before Phase7.
- Intraday screening runs Phase8 with the same intraday overlay storage used by Phase1/2/4, so it only computes the latest Alpha158 row.
- If the Phase8 model artifact is missing, both daily and intraday flows continue and write blank Phase8 fields.
- Main watchlist/focus keeps the current pattern + centered Top20 logic, then adds Phase8 Top5 as an extra source. Existing candidates receive a `p8_top5` source tag instead of duplicate rows.
- Final tables show `phase8_score_100`, `phase8_rank`, and `today_limit_up_excluded`.
- Markdown writing guides add P8 after `P4五日均/std` in the main reference table, plus a separate P8 Top5 table with P1/P2/P4/P5/P8, same-day return, ATR%, and recommended position.

## Fast Path

Phase8 prediction reuses the Phase4-style fast path:

```text
predict-limit-up-3d-opportunity --latest-only --feature-lookback-bars 61 --compact-output
```

Intraday uses the same latest-only logic on provisional intraday daily bars.

