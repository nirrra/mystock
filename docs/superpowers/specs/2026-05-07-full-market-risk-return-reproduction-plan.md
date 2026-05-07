# Full-Market Daily Risk and Return Model Reproduction Plan

Date: 2026-05-07

## 1. Context and Correction

The previous `event_risk_ranker` implementation used pattern matches as the model's sample universe. That does not match the new modeling direction. The next model family must be independent of pattern recognition and must use the full daily market panel:

```text
one sample = one symbol on one trade_date
```

Pattern recognition remains useful, but only as an evaluation slice and later combination signal. It must not be a prerequisite for training the risk model or return model.

This plan replaces the event-first design with a staged reproduction workflow. The first objective is to reproduce and validate a standalone daily risk model. The return model starts only after the risk model passes explicit out-of-sample gates.

## 2. Non-Negotiable Rules

- Use daily bars only. No minute data, tick data, or intraday reconstruction.
- Reproduce reference methods as closely as possible before adapting them.
- If an exact reproduction is blocked by data format, missing fields, unavailable history, incompatible package versions, or ambiguous paper details, pause implementation and discuss the adjustment.
- Evaluate risk, return, pattern, and combinations separately.
- Do not promote any model based on in-sample metrics.
- Do not integrate into `daily-screening` until the standalone reports show stable out-of-sample value.

## 3. Current Repository Implications

The repo already has local daily parquet data under `data/daily`, pattern scanners, daily screening outputs, and the earlier event ranker code. The new work should not delete the earlier event ranker, but it should not build on its pattern-event dataset.

The current local sample observed during the previous run starts around `2024-01-02` for at least `000001`. That is not enough to exactly reproduce papers using 8-10 year histories. Phase 0 must audit all symbols and report available history before any model training. If the usable full-market history cannot support the required lookbacks and walk-forward splits, stop and ask whether to fetch more data or relax the reproduction target.

## 4. Reference Methods and Reproduction Order

### 4.1 Method A: Tail-Decline Risk Classification

Primary source:

- Noh, S.-H. (2026). "Predicting Stock Market Risk Using Machine Learning Classification Models." `Risks`, 14(4), 92. https://www.mdpi.com/2227-9091/14/4/92

Original method details to reproduce first:

- Data: KOSPI 200 index daily values from 2015-01-01 to 2024-12-31.
- Return: daily log return `ln(S_t / S_{t-1})`.
- Label: a day is a risk event if its return falls below the 5th percentile of returns over the preceding 100 trading days.
- Split: 2015-2022 train, 2023-2024 validation. The paper also reports a 2025 test and a COVID stress-period test.
- Models compared: Logistic Regression, k-nearest Neighbor, Decision Tree, Random Forest, Linear Discriminant Analysis, Naive Bayes, Quadratic Discriminant Analysis, AdaBoost, and Gradient Boosting.
- Classification threshold: predicted probability above 0.5 means risk state.
- Metrics: accuracy, non-risk precision/recall/F1, risk precision/recall/F1, ROC AUC.
- Reported validation result: Logistic Regression is best by combined risk F1 and AUC.

Repo reproduction target:

1. First reproduce the exact index-level experiment on an A-share broad index if the repo has index history. Preferred index proxy is CSI 300 if available; otherwise use the broadest stable index available in local data.
2. Then adapt to the full-market stock panel by computing the same label per symbol and per date.
3. For stock-level modeling, start with the same nine classifiers. Logistic Regression is the primary baseline and must remain in every report.
4. Evaluate whether predicted high-risk buckets correspond to worse future returns and drawdowns.

Pause conditions:

- Pause if the repo lacks at least 900 trading days for the index-level reproduction.
- Pause if no index daily series exists and cannot be derived or fetched.
- Pause if only 2024-2026 data are available and the user has not approved a shorter, non-exact reproduction.

### 4.2 Method B: Triple-Barrier Risk Labeling

Primary sources:

- Lopez de Prado, M. (2018). `Advances in Financial Machine Learning`. Wiley.
- `nkonts/barrier-method`: https://github.com/nkonts/barrier-method
- Mlfin.py labeling documentation: https://mlfinpy.readthedocs.io/en/stable/Labelling.html

Original method details to reproduce first:

- Each observation has three barriers: upper profit-taking barrier, lower stop-loss barrier, and vertical time barrier.
- Label is determined by the first touched barrier.
- Upper barrier represents a positive return threshold; lower barrier represents a negative return threshold; vertical barrier is the maximum holding period.
- Common implementations scale horizontal barriers by volatility and produce labels in `+1, 0, -1`.
- Mlfin-style implementations often use daily volatility, event times, `pt_sl`, `min_ret`, vertical barriers, and optional side predictions for meta-labeling.

Repo reproduction target:

1. Implement a standalone daily label builder over the full market panel.
2. Use every eligible `symbol, trade_date` as an observation unless a later sample-thinning rule is explicitly approved.
3. Entry price is next trading day's open. This makes the sample tradable after observing `trade_date` close.
4. Use daily high/low after entry to determine whether the down barrier or up barrier is touched first.
5. Start with these grids:
   - horizon: `5, 10, 20` trading days
   - downside barrier: `1.0 * ATR14`, `1.5 * ATR14`, fixed `-5%`
   - upside barrier: optional for risk-only labeling; when used, `2.0 * ATR14` or fixed `+10%`
6. Risk label variants:
   - `barrier_down_first = 1` if lower barrier is touched before upper or vertical barrier.
   - `max_drawdown_exceed = 1` if future max drawdown exceeds the downside barrier even when the final return recovers.
7. For same-day high/low touching both barriers, use conservative `down_first` unless a later intraday data source is approved.

Pause conditions:

- Pause if the user wants exact mlfin-style event sampling by CUSUM rather than one observation per stock-date.
- Pause if A-share limit-up/limit-down execution rules materially change the label counts and need a stricter execution model.
- Pause if daily bars lack high/low/open for a significant share of symbols.

### 4.3 Method C: Qlib Alpha158 + LightGBM Framework

Primary sources:

- Microsoft Qlib paper: https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/
- Qlib benchmark README: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
- Qlib LightGBM Alpha158 config: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
- Qlib data documentation: https://qlib.readthedocs.io/en/latest/component/data.html
- Qlib Alpha158 handler source: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

Original method details to reproduce first:

- Qlib builds a daily panel dataset and predicts a score for all stocks each day.
- Alpha158 is a tabular hand-engineered technical feature set built from daily OHLCV expressions.
- Qlib's default Alpha158 label is `Ref($close, -2) / Ref($close, -1) - 1`, i.e. return from `T+1` to `T+2`, because a China stock can be bought on `T+1` after observing `T`.
- The benchmark LightGBM config uses:
  - market: `csi300`
  - benchmark: `SH000300`
  - train: 2008-2014
  - valid: 2015-2016
  - test/backtest: 2017-2020
  - model: `LGBModel`
  - loss: `mse`
  - learning_rate: `0.2`
  - max_depth: `8`
  - num_leaves: `210`
  - colsample_bytree: `0.8879`
  - subsample: `0.8789`
  - lambda_l1: `205.6999`
  - lambda_l2: `580.9768`
  - portfolio strategy: `TopkDropoutStrategy(topk=50, n_drop=5)`
  - exchange assumptions: limit threshold `0.095`, close deal price, open cost `0.0005`, close cost `0.0015`, min cost `5`.
- Qlib evaluates signal quality by IC, ICIR, RankIC, RankICIR and portfolio quality by annualized return, information ratio, and max drawdown.

Repo reproduction target:

1. Try to run Qlib as a reference implementation first, using local data converted to Qlib-compatible format if feasible.
2. If full Qlib integration is blocked, implement an Alpha158-compatible local feature builder and document every deviation from Qlib expressions.
3. Use Qlib-style labels for the return model only after the risk model has passed.
4. For risk modeling, reuse Alpha158 features with the best risk label from Method A or B and train a LightGBM classifier.
5. For return modeling, reproduce LightGBM regression on Qlib-style next tradable return.

Pause conditions:

- Pause if Qlib installation or data conversion requires nontrivial package changes or external downloads.
- Pause if adjusted price semantics differ from Qlib's adjusted `close` enough to change labels materially.
- Pause if the local data does not have enough history to support Qlib-style train/valid/test splits.

### 4.4 Method D: MCD Crash-Risk Label

Primary source:

- Karasan, A., Alp, O. S., and Weber, G.-W. (2025). "Machine learning approach to stock price crash risk." `Annals of Operations Research`, 350, 1053-1074. https://link.springer.com/article/10.1007/s10479-025-06596-7

Original method details to reproduce first:

- Data: North American NYSE/NASDAQ firms from CRSP and Compustat.
- Frequency: weekly stock returns.
- The paper detects anomalies in firm-specific stock returns using Minimum Covariance Determinant.
- The dependent variable `NEGOUTLIER` is 1 for negative outliers classified as crash events.
- The paper compares this MCD-based crash measure with traditional crash-risk measures such as `CRASH`, `NCSKEW`, and `DUVOL`.
- The predictive analysis uses cross-sectional or panel regression with firm-specific investor sentiment and controls.

Repo reproduction target:

1. Reproduce the label-generation part first, not the full corporate finance regression.
2. Convert daily bars to weekly returns.
3. Estimate firm-specific returns by removing market return when a market index is available.
4. Apply robust covariance / MCD anomaly detection to identify negative outliers.
5. Compare MCD labels with simpler crash labels:
   - weekly return below `mean - 3.2 * std`
   - negative skewness proxy
   - down-to-up volatility proxy
6. Use MCD crash label as a supplemental extreme-risk target, not as the main short-horizon trading risk label unless it passes out-of-sample gates.

Pause conditions:

- Pause if the user expects full paper reproduction including Compustat accounting controls or investor sentiment. The current repo does not appear to contain those fields.
- Pause if weekly market-adjusted returns cannot be built reliably.
- Pause if MCD labels are too sparse for the available local history.

### 4.5 Method E: MASTER / Transformer Return Model

Primary sources:

- Li et al. (2024). "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting." AAAI 2024. https://huggingface.co/papers/2312.15235
- Official code: https://github.com/SJTU-DMTai/MASTER

Original method details to reproduce first:

- MASTER models momentary and cross-time stock correlations.
- It uses market information to guide feature selection.
- Official released data are grouped by prediction date with shape `(N, T, F)`.
- Reported configuration includes lookback window `T=8`.
- Feature dimension in the official README is 222: 158 stock factors, 63 market information features, and 1 label.
- Preprocessing includes robust z-score normalization using median and MAD, clipping to `[-3, 3]`, fillna, cross-sectional z-score label normalization, and dropping 5% extreme labels during training.
- The official repository warns about validation-data processing issues in earlier dumps and provides notes for corrected usage.

Repo reproduction target:

1. Do not start MASTER until the LightGBM return model has produced useful out-of-sample IC/RankIC or portfolio uplift.
2. First run the official repo on its published data if dependencies and downloads are available.
3. Then build a local `(date, symbol, lookback, feature)` tensor from Alpha158-style features and market features.
4. Compare against LightGBM on the same train/valid/test dates.

Pause conditions:

- Pause if the official data download is unavailable.
- Pause if PyTorch/Qlib dependency versions conflict with the repo environment.
- Pause if the local data cannot produce the market information fields required by MASTER.
- Pause if LightGBM fails; do not use MASTER as a complexity-first rescue.

## 5. Implementation Phases

### Phase 0: Data Audit and Shared Panel

Deliverables:

- `reports/full_market_model/data_audit.csv`
- `reports/full_market_model/data_audit_summary.json`
- Shared panel builder design for `symbol, trade_date` rows.

Audit fields:

- symbol
- first_trade_date
- last_trade_date
- trading_days
- missing open/high/low/close/volume/amount counts
- limit-up/limit-down detectable count
- eligibility by horizon and lookback

Gate:

- Continue only if there is enough history for the selected exact reproduction. If not, pause and present choices: fetch more daily data, use shorter non-exact splits, or limit reproduction to the methods that fit available data.

### Phase 1: Tail-Decline Risk Reproduction

Deliverables:

- `reports/full_market_model/tail_risk_index_reproduction.csv`
- `reports/full_market_model/tail_risk_panel_metrics.csv`
- `reports/full_market_model/tail_risk_decile_report.csv`

Steps:

1. Build log returns.
2. Build rolling 100-day 5th percentile labels.
3. Reproduce index-level classifiers first.
4. Adapt to per-symbol panel labels.
5. Train the nine classifiers from the paper.
6. Evaluate out-of-sample by risk classification metrics and trading-impact metrics.

Risk model pass criteria:

- Risk PR-AUC beats the unconditional risk-rate baseline in at least 70% of walk-forward windows.
- Top predicted-risk decile has worse future downside metrics than bottom predicted-risk decile in at least 70% of windows.
- Filtering the highest predicted-risk 20% improves full-market future max drawdown and does not materially worsen average return.
- On the pattern subset, `pattern + risk_filter` reduces stop/downside events versus `pattern_only` without eliminating more than 60% of pattern candidates.

### Phase 2: Triple-Barrier Risk Reproduction

Deliverables:

- `reports/full_market_model/barrier_label_distribution.csv`
- `reports/full_market_model/barrier_risk_metrics.csv`
- `reports/full_market_model/barrier_vs_tail_comparison.csv`

Steps:

1. Build daily full-market triple-barrier labels.
2. Validate label distributions by horizon and barrier setting.
3. Train Logistic Regression and LightGBM classifiers.
4. Compare against tail-decline labels.
5. Select the risk target with better out-of-sample filtering behavior.

Gate:

- Continue only if one risk target has stable out-of-sample value. If neither target passes, stop and diagnose labels/features before adding return prediction.

### Phase 3: Qlib Alpha158 + LightGBM Risk Model

Deliverables:

- `reports/full_market_model/alpha158_feature_audit.csv`
- `reports/full_market_model/alpha158_risk_metrics.csv`
- model artifact under `data/ml/full_market_risk/`

Steps:

1. Attempt Qlib-based reproduction or document why local Alpha158-compatible implementation is required.
2. Build Alpha158-style daily features.
3. Apply leakage-safe train-fit normalization.
4. Train LightGBM classifier using the selected risk label.
5. Compare against Phase 1 and Phase 2 models.

Gate:

- Promote to return modeling only if Alpha158+LightGBM beats simpler baselines on out-of-sample risk filtering and is not purely overfit.

### Phase 4: Qlib Alpha158 + LightGBM Return Model

Deliverables:

- `reports/full_market_model/return_model_signal_metrics.csv`
- `reports/full_market_model/return_model_portfolio_metrics.csv`
- `reports/full_market_model/risk_return_combo_metrics.csv`

Steps:

1. Use Qlib-style label `T+1 to T+2` return first.
2. Add `5d`, `10d`, and `20d` forward return labels only after the one-day tradable label is working.
3. Train LightGBM regression/ranking models.
4. Evaluate IC, RankIC, decile return spread, portfolio max drawdown, and turnover.
5. Combine with risk model:
   - return only
   - risk only
   - risk then return
   - pattern only
   - pattern + risk
   - pattern + return
   - pattern + risk + return

Gate:

- Continue only if return model contributes incremental value after risk filtering.

### Phase 5: MCD Crash-Risk Supplemental Label

Deliverables:

- `reports/full_market_model/mcd_crash_label_report.csv`
- `reports/full_market_model/crash_label_comparison.csv`

Steps:

1. Convert daily data to weekly data.
2. Build market-adjusted firm-specific returns.
3. Generate MCD negative outlier labels.
4. Compare with tail-decline and triple-barrier labels.
5. Use as a stress-risk auxiliary output if it adds signal.

Gate:

- Do not replace the main short-horizon risk model unless MCD improves short-horizon trading risk metrics.

### Phase 6: MASTER Reproduction

Deliverables:

- `reports/full_market_model/master_official_reproduction.md`
- `reports/full_market_model/master_local_comparison.csv`

Steps:

1. Run the official MASTER repo on its released data if possible.
2. Rebuild local tensor data only after official run is understood.
3. Compare MASTER to LightGBM on identical labels and splits.
4. Use MASTER only if it beats LightGBM out-of-sample after costs and drawdown checks.

Gate:

- Do not implement or tune MASTER until the simpler return model has passed.

## 6. Evaluation Framework

Every method must report these categories.

Risk classification metrics:

- ROC AUC
- PR-AUC
- risk precision, recall, F1
- confusion matrix at fixed threshold `0.5`
- threshold sweep by target coverage
- calibration curve and Brier score

Risk trading-impact metrics:

- future mean return by predicted-risk decile
- future max drawdown by predicted-risk decile
- downside-barrier hit rate by predicted-risk decile
- filter highest-risk 10/20/30% impact on remaining universe
- same reports on pattern-only subset

Return signal metrics:

- IC
- RankIC
- decile spread
- top/bottom bucket forward return
- turnover

Portfolio-style metrics:

- average return
- annualized return when applicable
- max drawdown
- information ratio
- hit rate
- profit factor
- coverage days

Combination reports:

- `market_all`
- `risk_only`
- `return_only`
- `pattern_only`
- `pattern_plus_risk`
- `pattern_plus_return`
- `pattern_plus_risk_plus_return`

## 7. Data Leakage Rules

- Features for `trade_date = T` may only use data available at or before T close.
- Tradable return labels must assume first possible action is T+1.
- Normalizers are fit only on train windows.
- Rolling labels must use only past data for thresholds.
- Walk-forward test windows must be separated from training by an embargo at least equal to the maximum forward horizon.
- Survivorship bias must be reported. If the repo's universe is current-listing only, the report must mark this limitation.

## 8. Proposed Module Boundaries

The implementation should add new modules rather than modify the earlier event ranker:

- `src/stocks_analyzer/full_market_panel.py`
- `src/stocks_analyzer/full_market_labels.py`
- `src/stocks_analyzer/full_market_features.py`
- `src/stocks_analyzer/full_market_risk.py`
- `src/stocks_analyzer/full_market_return.py`
- `src/stocks_analyzer/full_market_evaluation.py`

Suggested CLI commands:

- `audit-full-market-data`
- `build-full-market-panel`
- `reproduce-tail-risk`
- `reproduce-barrier-risk`
- `train-full-market-risk`
- `validate-full-market-risk`
- `train-full-market-return`
- `validate-full-market-return`
- `evaluate-model-combinations`

## 9. Promotion Standard

A risk model can move to the return-model stage only if:

- It passes the risk model pass criteria in Phase 1 or Phase 2.
- It beats a no-model baseline and a simple rule baseline.
- It improves the pattern subset without depending on pattern as an input.
- Its benefit appears in multiple walk-forward windows, not just one window.

A return model can move to combination testing only if:

- RankIC is positive in most windows.
- Top score buckets outperform bottom buckets after realistic tradability assumptions.
- It still adds value after excluding high-risk names.

A combined model can be considered for daily-screening integration only if:

- `pattern + risk + return` beats `pattern_only`.
- `risk + return` has standalone value outside pattern.
- Drawdown and downside-hit rates improve, not only average return.
- Generated watchlists are sparse enough to trade and broad enough to avoid single-theme overfit.

## 10. Research References

- Noh, S.-H. (2026). "Predicting Stock Market Risk Using Machine Learning Classification Models." `Risks`, 14(4), 92. https://www.mdpi.com/2227-9091/14/4/92
- Yang, X., Liu, W., Zhou, D., Bian, J., and Liu, T.-Y. (2020). "Qlib: An AI-oriented Quantitative Investment Platform." Microsoft Research / arXiv. https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/
- Microsoft Qlib benchmark README. https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
- Microsoft Qlib LightGBM Alpha158 config. https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
- Microsoft Qlib data documentation. https://qlib.readthedocs.io/en/latest/component/data.html
- Lopez de Prado, M. (2018). `Advances in Financial Machine Learning`. Wiley.
- `nkonts/barrier-method`. https://github.com/nkonts/barrier-method
- Mlfin.py labeling documentation. https://mlfinpy.readthedocs.io/en/stable/Labelling.html
- Karasan, A., Alp, O. S., and Weber, G.-W. (2025). "Machine learning approach to stock price crash risk." `Annals of Operations Research`, 350, 1053-1074. https://link.springer.com/article/10.1007/s10479-025-06596-7
- Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H., and Huang, S. (2024). "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting." AAAI 2024. https://huggingface.co/papers/2312.15235
- Official MASTER repository. https://github.com/SJTU-DMTai/MASTER
