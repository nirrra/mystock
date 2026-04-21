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
- The day volume is at least `3x` the average volume of the previous `20` trading days.
- The `20`-day average volume used above excludes the breakout day itself.

No breakout-day close-quality rule is imposed. The breakout may close back below the old-high price and still count as a valid breakout event if the above conditions are satisfied.

## New Pattern Definitions

### Pattern1: Pre-Breakout Watch

Purpose: find stocks still below the old high but close enough that the next day may produce a valid breakout.

Hard conditions:

- A valid mother-pattern old high exists.
- No valid breakout day has occurred after that old high and up to the current analysis day.
- Current close is below or equal to the old-high price.
- Current close is within `8%` of the old-high price.

Interpretation:

- This is the "临门一脚" setup.
- The stock is still under resistance, but the distance is tight enough for next-day monitoring.

### Pattern2: Breakout Confirmation

Purpose: find stocks whose latest completed daily bar is itself the valid breakout day.

Hard conditions:

- A valid mother-pattern old high exists.
- The latest completed daily bar satisfies the full breakout-day definition against that old high.

Interpretation:

- This is the first completed-bar confirmation that `量顶天立地` has actually triggered.
- It is a next-day follow-up candidate, not a same-day chase signal.

### Pattern3: Post-Breakout Follow-Through

Purpose: find stocks within `1` to `8` trading days after a valid breakout that remain in a reasonable next-day ambush zone.

Hard conditions:

- A valid mother-pattern old high exists.
- A valid breakout day exists after that old high.
- The current analysis day is `1` to `8` trading days after the breakout day.
- Current close is no more than `10%` above the old-high price.

Pattern3 includes two accepted sub-cases:

1. Direct continuation:
   The stock continues higher after breakout, but current close is still within the allowed `10%` extension limit above the old-high price.
2. MA20 pullback recovery:
   The stock may intraday break below `MA20`, but the current close must recover back above `MA20`.

Interpretation:

- This pattern covers both immediate follow-through and controlled pullback re-entry.
- The old-high price, not the breakout-day close, is the reference level for the `10%` extension cap.

## Pattern Number Migration

The visible pattern numbering should change as follows:

- new `pattern1` = `量顶天立地` pre-breakout watch
- new `pattern2` = `量顶天立地` breakout confirmation
- new `pattern3` = `量顶天立地` post-breakout follow-through
- old `pattern2` becomes new `pattern4`
- old `pattern3` becomes new `pattern5`
- old `pattern4` becomes new `pattern6`

Internal strategy names should be decoupled from display numbering so future renumbering does not force another broad refactor.

## Detection Order And Overlap Rules

The system should evaluate the shared mother-pattern structure first, then classify into the three new patterns.

Expected precedence:

- If the latest bar is a valid breakout day, classify as `pattern2`.
- Else if a valid breakout day exists within the prior `1-8` trading days and the stock is still in range, classify as `pattern3`.
- Else if no breakout has happened yet and price is within `8%` below the old high, classify as `pattern1`.

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
- breakout-day volume ratio versus the prior `20`-day average
- pattern reason string describing the matched stage

The output should make it obvious which historical peak was chosen and whether the stock is pre-breakout, on-breakout, or post-breakout.

## Non-Goals

- No intraday confirmation logic
- No scoring model for bottom quality
- No additional close-quality constraint on the breakout day
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
- breakout day requires bullish candle, high above old high, and `3x` prior-20-day average volume
- `pattern1` matches near-old-high candidates within `8%` below resistance
- `pattern2` matches only when the latest bar is the valid breakout day
- `pattern3` matches only within `1-8` bars after breakout and rejects extensions above `10%`
- `pattern3` accepts MA20 intraday breaks only when the close recovers above `MA20`
- numbering, CLI selection, and report labels reflect the new `1-6` order

## Open Assumptions Resolved

The following design decisions are fixed for implementation unless later changed explicitly:

- bottoming structure is a hard requirement
- bottoming structure does not need a flat box and may resemble a `黄金坑`
- old high is the most recent qualifying peak, not the highest historical high
- breakout uses day high crossing the old-high price
- breakout candle must be bullish
- breakout volume uses prior-20-day average volume, excluding the breakout day
- `pattern3` uses old-high price as the extension reference
- `pattern3` includes both direct continuation and MA20 pullback recovery
