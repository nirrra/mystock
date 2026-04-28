# Volume Top Breakout Pattern Redesign

## Goal

Redesign the pattern system around a shared `量顶天立地` mother pattern so the nightly review flow can prepare next-day ambush candidates using only end-of-day daily bars.

## Background

The current `type1` logic partially overlaps with the intended `量顶天立地` idea, but it only captures "near old high" candidates and does not model the full lifecycle:

- pre-breakout watch
- breakout day confirmation
- post-breakout continuation or pullback

The redesign should define one shared structural event first, then derive the new `pattern1`, `pattern2`, and `pattern3` from that event. The old `pattern2`, `pattern3`, and `pattern4` should shift to `pattern4`, `pattern5`, and `pattern6`.

## Trading Constraint

The system only has access to completed daily bars up to the most recent trading day. It cannot react intraday. All detections therefore support next-day planning rather than same-session execution.

## Shared Mother Pattern

### Intent

`量顶天立地` here means a stock that formed a meaningful historical swing peak, spent a long enough period in a repaired bottoming structure after that peak, and either is close to breaking that peak or has already broken it with clear volume expansion.

### Old-High Definition

An old high is valid only when all of the following are true:

- It is the most recent qualifying peak.
- It is a local peak within a symmetric `21`-bar window: the peak day high must be the highest high within `10` trading days on each side.
- The peak day is at least `60` trading days before the current analysis day.
- After the peak, the stock later experienced at least `10%` drawdown from that peak high to a subsequent low.
- After the peak, price fell below `MA60` at least once before the current analysis day.

This definition intentionally treats the old high as a meaningful prior swing peak rather than the absolute highest price in a long lookback window.

### Breakout-Day Definition

A breakout day is valid only when all of the following are true:

- The day is a bullish candle: `close > open`.
- The day high is above the selected old-high price.
- The day volume is greater than the highest volume of the previous `90` trading days.
- The `90`-day volume-high window excludes the breakout day itself.
- The candle close position is at least `0.60`: `(close - low) / (high - low) >= 0.60`.
- The upper shadow is no more than `0.35` of the full candle range.
- The real body is at least `0.25` of the full candle range.
- Turnover is exported as an auxiliary risk field, but is not a hard filter in this phase.

The breakout day must show acceptable candle quality. It may still close below the old-high price and count as a valid breakout event if the candle-quality, volume, and high-crossing rules are all satisfied.

## New Pattern Definitions

### Pattern1: Pre-Breakout Watch

Purpose: find stocks still below the old high but close enough that the next day may produce a valid breakout.

Hard conditions:

- A valid mother-pattern old high exists.
- No valid breakout day has occurred after that old high and up to the current analysis day.
- Current close is below or equal to the old-high price.
- Current close is within `5%` of the old-high price.

Interpretation:

- This is the "临门一脚" setup.
- The stock is still under resistance, but the distance is tight enough for next-day monitoring.

### Pattern2: Price Above Old High

Purpose: find stocks that have already produced a valid volume breakout and whose current completed daily bar closes above the old-high price.

Hard conditions:

- A valid mother-pattern old high exists.
- A valid breakout day exists after that old high.
- The current analysis day is `1` to `10` trading days after the breakout day.
- Current close is above the old-high price.
- From the breakout day through the current analysis day, every close is at or above `MA20 * 0.98`.

Interpretation:

- This is the stronger post-breakout state: price is already above the prior resistance while still respecting the MA20 floor.
- It is a next-day follow-up candidate, not a same-day chase signal.

### Pattern3: Pullback Below Old High Above MA20

Purpose: find stocks within `1` to `10` trading days after a valid breakout that remain in a reasonable next-day ambush zone.

Hard conditions:

- A valid mother-pattern old high exists.
- A valid breakout day exists after that old high.
- The current analysis day is `1` to `10` trading days after the breakout day.
- Current close is below the old-high price.
- Current close is at or above `MA20 * 0.98`.
- From the breakout day through the current analysis day, every close is at or above `MA20 * 0.98`.
- Current-day volume is below the current `5`-day average volume.

Interpretation:

- This pattern covers a controlled, shrinking-volume pullback after the valid volume breakout.
- The old-high price and MA20 floor define the active observation zone.

## Pattern Number Migration

The visible pattern numbering should change as follows:

- new `pattern1` = `量顶天立地` pre-breakout watch
- new `pattern2` = `量顶天立地` post-breakout price above old high
- new `pattern3` = `量顶天立地` shrinking-volume pullback above MA20 floor
- old `pattern2` becomes new `pattern4`
- old `pattern3` becomes new `pattern5`
- old `pattern4` becomes new `pattern6`

Internal strategy names should be decoupled from display numbering so future renumbering does not force another broad refactor.

## Detection Order And Overlap Rules

The system should evaluate the shared mother-pattern structure first, then classify into the three new patterns.

Expected precedence:

- If a valid breakout exists within the prior `1-10` trading days and current close is above the old-high price while the MA20 floor has held, classify as `pattern2`.
- Else if a valid breakout exists within the prior `1-10` trading days, current close is below the old high but above `MA20 * 0.98`, and current-day volume is below the current `5`-day average volume, classify as `pattern3`.
- Else if no breakout has happened yet and price is within `5%` below the old high, classify as `pattern1`.

This keeps the three patterns mutually understandable and avoids duplicate labeling for the same lifecycle stage.

## Output Expectations

Each matched row should preserve the fields needed for review and plotting:

- selected old-high date
- selected old-high price
- days since old high
- breakout date when applicable
- current close
- distance to old high
- current extension above old high when applicable
- breakout-day volume ratio versus the prior `90`-trading-day volume high
- breakout-day close position, upper-shadow ratio, real-body ratio
- breakout-day turnover and turnover state
- pattern reason string describing the matched stage

The output should make it obvious which historical peak was chosen and whether the stock is pre-breakout, on-breakout, or post-breakout.

## Non-Goals

- No intraday confirmation logic
- No scoring model for bottom quality
- No turnover hard filter on the breakout day
- No requirement that the old high be the absolute highest price in a broader long-term window
- No redesign of unrelated trend, MACD, ATR, or watchlist ranking logic in this phase

## Implementation Outline

1. Extract a shared `量顶天立地` detector that selects the most recent qualifying old high and optionally the associated breakout day.
2. Build new strategy evaluators for the new `pattern1`, `pattern2`, and `pattern3` on top of that shared detector.
3. Rename or remap the existing `pattern2`, `pattern3`, and `pattern4` to `pattern4`, `pattern5`, and `pattern6`.
4. Update CLI flags, label maps, reports, and watchlist references so user-facing numbering matches the new design.
5. Add tests that verify old-high selection, breakout-day qualification, and stage classification.

## Testing

At minimum, tests should cover:

- old-high selection chooses the most recent qualifying local peak
- peaks that do not have `60` trading days of separation are rejected
- peaks without at least `10%` drawdown are rejected
- peaks without an intervening drop below `MA60` are rejected
- breakout day requires bullish candle, high above old high, a volume high versus the prior `90` trading days, and acceptable candle quality
- `pattern1` matches near-old-high candidates within `5%` below resistance
- `pattern2` matches only within `1-10` bars after breakout when current close is above the old high and all post-breakout closes hold above `MA20 * 0.98`
- `pattern3` matches only within `1-10` bars after breakout when current close is below the old high but above `MA20 * 0.98`
- `pattern3` rejects any post-breakout close below `MA20 * 0.98`
- `pattern3` rejects pullbacks whose current-day volume is not below the current `5`-day average volume
- numbering, CLI selection, and report labels reflect the new `1-6` order

## Open Assumptions Resolved

The following design decisions are fixed for implementation unless later changed explicitly:

- bottoming structure is a hard requirement
- bottoming structure does not need a flat box and may resemble a `黄金坑`
- old high is the most recent qualifying peak, not the highest historical high
- breakout uses day high crossing the old-high price
- breakout candle must be bullish
- breakout volume must exceed the prior-90-trading-day volume high, excluding the breakout day
- `pattern2` uses the old-high price as the above-resistance reference
- `pattern3` uses the old-high price and `MA20 * 0.98` as its pullback zone
