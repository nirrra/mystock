# V4.1 Long Quality Ranker Design

## Context

V4 split the model into a risk filter and a long-upside ranker. The risk filter worked: validation/test risk AUC stayed near 0.70, and the hard gate reduced bad-risk rate, stop-loss rate, and average drawdown. The long-upside ranker did not work well enough: within the risk-passed candidate pool, test-set rank correlation between `long_upside_score` and realized long-upside value was close to zero.

The likely cause is target and loss mismatch. V4 trained a pointwise regressor on all samples, then used it as a daily TopN ranker inside a much narrower low-risk candidate pool. It also used long stage1 up/down classification outputs, which did not add much new revenue-quality signal beyond the existing 20/60-day classifiers.

## Goal

V4.1 should train a dedicated long-quality ranker that answers:

> Among stocks already passing the V4 risk gate on the same trade date, which ones have the best future 20/60-day return quality?

It is not responsible for risk filtering. Risk remains a hard precondition handled by V4 risk score and down-probability caps.

## Label

The training label is a daily cross-sectional grade computed only within the V4 risk-passed candidate pool.

First compute horizon quality:

```text
q20 =
  0.45 * clip(period_return_20d / 0.15, -1.0, 1.5)
+ 0.25 * clip(max_upside_20d / 0.15, 0.0, 1.5)
+ 0.15 * clip(period_return_20d / max_upside_20d, -1.0, 1.0)
- 0.15 * clip(max_drawdown_20d / 0.08, 0.0, 1.5)

q60 =
  0.50 * clip(period_return_60d / 0.30, -1.0, 1.5)
+ 0.25 * clip(max_upside_60d / 0.30, 0.0, 1.5)
+ 0.15 * trend_persistence_60
- 0.10 * clip(max_drawdown_60d / 0.15, 0.0, 1.5)

long_quality = 0.4 * q20 + 0.6 * q60
```

`trend_persistence_60` is a bounded 60-day persistence term derived from positive final return, realized upside capture, and the 20-day confirmation return. It rewards sustained moves rather than one-day spikes.

Then rank `long_quality` inside each `trade_date` candidate pool and convert to grades:

```text
grade 4: top 10%
grade 3: top 10%-25%
grade 2: middle 35%
grade 1: bottom 15%-40%
grade 0: bottom 15%
```

This makes the label relative to the daily opportunity set and avoids bull/bear regime leakage into an absolute return target.

## Model And Loss

Primary model:

```text
LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[20, 50])
group = trade_date
label = long_quality_grade
label_gain = [0, 1, 3, 7, 15]
```

Fallback model when LightGBM is unavailable or the sample is too small:

```text
HistGradientBoostingRegressor
target = daily long_quality rank percentile
sample_weight = higher for grade 0, 3, and 4
```

The fallback is acceptable for tests and small samples, but the main production path should prefer LambdaRank because the live objective is daily TopN ordering.

## Training Flow

1. Build the existing stacked dataset and V4 risk/upside labels.
2. Generate walk-forward OOF risk stage1 predictions on the training split.
3. Train the V4 risk filter on OOF risk features.
4. Score train/valid/test with the risk model.
5. Apply the selected V4 risk gate to each split.
6. Within the risk-passed training pool, compute `long_quality`, daily rank percentiles, and grades.
7. Train the V4.1 long-quality ranker on the risk-passed training pool only.
8. Score valid/test risk-passed pools and evaluate ranking quality and TopN outcomes.
9. For deployment, refit risk stage1/risk filter on the full dataset, build the risk-passed pool, then train the deployment long-quality ranker on that pool.

## Features

V4.1 uses features that can support relative long-horizon return quality:

- V4 risk context: `risk_score`, 20/60 down probabilities, weighted down probability.
- 20/60 up/down probabilities as supporting signals, not as the whole model.
- 20/40/60/120-day return, range position, MA20/MA60 structure, and MA slopes.
- 20/60-day block returns, block volume/amount sums, and block change ratios.
- MACD, RSI, KDJ, ADX, Bollinger, ATR technical features.
- Daily cross-sectional ranks for core numeric features where available.

Short 5/10-day return can be used as an overheat/context feature, but the label and model objective do not directly reward 5/10-day outcomes.

## Evaluation

Evaluate the long-quality ranker separately from the risk filter:

- Candidate-pool Rank IC: Pearson/Spearman between score and `long_quality`.
- NDCG@20 and NDCG@50.
- Score decile monotonicity: average future return, long_quality, stop-loss rate, and drawdown by score bucket.
- Top20/Top50 vs candidate-pool average:
  - win rate
  - 20/60-day average return
  - 20/60-day take-profit rate
  - 20/60-day stop-loss rate
  - average drawdown
  - bad-risk rate

V4.1 is successful only if TopN improves over the candidate-pool average and score buckets show reasonable monotonicity on validation and test. A lower stop-loss rate alone is not enough because that belongs to the risk filter.

## Non-Goals

- Do not replace the V4 risk filter.
- Do not optimize for 5/10-day profit targets.
- Do not judge the ranker only by all-sample MAE/RMSE.
- Do not treat the combined V4.1 trading pipeline as the only metric; risk and return ranking must remain separately reported.
