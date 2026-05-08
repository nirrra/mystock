# Task Plan: A股技术面分析框架 V1

## Active Initiative: 2026-05-08 Mainline and Picks Refresh

### Goal

根据本地 `reports/xueqiu` 最新博主观点、最新主 `watchlist_2026-05-08.json` 和 `track_stock.xlsx` 的 `Sheet2`，更新 `主线.md` 与 `选股.md`。同日旧选股段落必须整段替换，不沿用旧列表。

### Implementation Phases

- [x] 确认最新雪球观点文件和最新 watchlist 日期。
- [x] 读取 `docs/picks-writing-guide.md` 与 `track_stock.xlsx` Sheet2。
- [x] 更新 `主线.md` 的最新主线口径。
- [x] 新增 `选股.md` 中 `2026.5.8` 段落，并保留新到旧排序。
- [x] 校验日期排序、固定清单和 Phase/ATR 字段完整性。
- **Status:** complete

## Goal

构建一个面向 A 股主板的 Python 命令行技术面分析框架，先基于免费数据源跑通股票池、日线缓存、基础指标、三类参数化模板和可解释结果输出。

## Current Phase

Daily watchlist semantics have been adjusted. The current daily flow still runs full-market `update -> tradingview -> predict-model -> macd -> atr -> pattern`, but `trade_permission` is now a next-open warning rather than a hard entry gate. The main watchlist combines low-risk model Top20 names with low-risk pattern matches.

V5 and V5.1 model iterations are complete as experimental research paths. V5 adds daily 1/5/20 volume-price risk and quality signals; V5.1 adds a candidate-only ranker on top of the already filtered candidate pool. Neither is promoted to `predict-model` because validation/test Top20 trading metrics do not consistently beat the current `v42_gate_v4_rank` daily model.

## Active Initiative: Daily Watchlist Semantics

### Goal

Always produce a useful daily candidate list while preserving the model's next-open no-trade warning.

### Implementation Phases

- [x] Keep full-market daily stages unchanged.
- [x] Make risk filtering the hard watchlist entry gate.
- [x] Treat `trade_permission` as a top-level next-open warning.
- [x] Build main `watchlist` from low-risk model Top20 plus low-risk pattern matches.
- [x] Keep `watchlist_pattern` as low-risk pattern subset.
- [x] Update pick-writing guide and README.
- [x] Add watchlist regression tests.
- **Status:** complete

The active validation task is now lightweight walk-forward testing for the current mainline `v42_gate_v4_rank`, so the single-split 80% test win rate can be checked across multiple future windows.

## Active Initiative: Lightweight Walk-Forward Validation

### Goal

Validate whether the current mainline model generalizes across multiple chronological market windows before treating its 80% single-split Top20 win rate as stable evidence.

### Implementation Phases

- [x] Write walk-forward validation design spec.
- [x] Add chronological trading-date window generation.
- [x] Add in-memory V4.2 hybrid retraining per window without overwriting daily model artifacts.
- [x] Add per-window TopN reports and aggregate summary thresholds.
- [x] Wire CLI command `validate-model-walkforward`.
- [x] Add focused unit tests for window generation, summary aggregation, and CLI parsing.
- [ ] Run a full multi-window local validation job and record the resulting stability metrics.
- **Status:** implemented, awaiting full run

## Active Initiative: V5.1 Candidate-Only Ranker

### Goal

Improve the unresolved stock-selection problem after the current risk/opportunity gates have already filtered the market: among allowed candidates, learn which stocks deserve higher buy priority.

### Implementation Phases

- [x] Add candidate-only rank labels focused on 20-day return quality with 60-day support.
- [x] Add cross-sectional candidate-pool rank features.
- [x] Train a V5.1 ranker using LightGBM LambdaRank when available, with a histogram-boosting fallback.
- [x] Blend V5.1 ranker score with the existing V4.2 rank score using validation Top20 selection.
- [x] Add artifact save/load, prediction reports, comparison metrics, CLI commands, and tests.
- [x] Run full local training and compare V4.2, V5, and V5.1.
- **Status:** complete, experimental; not promoted

### Latest V5.1 Result

- Trained full local dataset through `train-candidate-ranker --max-iter 40 --top-n 20,50 --predict-date 2026-04-30`.
- Artifact: `data/ml/v51_candidate_ranker/v51_candidate_ranker.pkl`.
- Reports: `reports/v51_candidate_ranker/`.
- Selected blend: `0.65 * V5.1 ranker + 0.35 * V4.2 rank score`.
- V5.1 improved rank diagnostics but degraded Top20 trading quality versus V5 and did not beat V4.2 on validation, so it is not eligible for daily promotion.

## Active Initiative: V5 Volume-Price Fusion

### Goal

Implement the approved V5 design: keep the V4.2 opportunity gate and V4 risk/upside base, add multi-period daily volume-price risk and quality signals, then learn a fusion score that improves Top20 return quality without materially worsening stop-loss or bad-risk metrics.

### Implementation Phases

- [x] Inspect current V4.2 training/prediction functions and reusable helper boundaries.
- [x] Add testable 1d/5d/20d volume-price feature generation.
- [x] Add volume-price extreme risk flag, risk label, and healthy-path quality target.
- [x] Train OOF volume-price risk and quality submodels.
- [x] Add V5 fusion training, prediction reporting, artifact save/load, and comparison metrics.
- [x] Wire CLI commands and keep `predict-model` promotion guarded until V5 beats V4.2 hybrid.
- [x] Add focused tests and run targeted verification.
- **Status:** complete, experimental; not promoted

### Latest V5 Result

- Trained full local dataset through `train-volume-price-fusion --reuse-base-artifact --max-iter 40 --top-n 20,50 --predict-date 2026-04-30`.
- Artifact: `data/ml/v5_volume_price_fusion/v5_volume_price_fusion.pkl`.
- Reports: `reports/v5_volume_price_fusion/`.
- Baseline mode: reused existing V4.2 hybrid artifact for the base layer; the stricter retrain-base path is implemented but was too slow at `max_iter=80` for the current full dataset.
- Test Top20 improved slightly versus `v42_gate_v4_rank`; validation Top20/Top50 did not, so V5 is not yet eligible for daily promotion.

## Active Initiative: Project Code Cleanup and Documentation Sync

### Goal

Remove obsolete strong-candidate and medium-candidate code paths, keep only the current daily-screening model flow, and update local project documentation so future work resumes from the correct architecture.

### Cleanup / Documentation Phases

- [x] Delete obsolete probability-model modules, old V3/V3.1/V4 wrapper entry points, `runcmd`, plotting, and Xueqiu archive code.
- [x] Remove tests that only covered deleted modules.
- [x] Keep current model flow: `train-opportunity-ranker`, `predict-opportunity-ranker`, `predict-model`, `daily-screening`, `watchlist`.
- [x] Verify core compile and targeted tests.
- [x] Update README, findings, progress, task plan, and picks writing guide.
- **Status:** complete

## Active Initiative: V4.2 Opportunity-Gated Ranker

### Goal

Implement V4.2 as a two-step model: reuse the V4 risk filter, add a date-level opportunity gate that can choose no-trade days, then train a conditional stock ranker only on historical good-opportunity days.

### Implementation Phases

- [x] Inspect current V4/V4.1 training, prediction, report, and test structure.
- [x] Add V4.2 result dataclass, dirs, labels, opportunity aggregation, model fitting, threshold selection, and scoring utilities.
- [x] Add V4.2 train/predict workflow and model artifact persistence.
- [x] Wire CLI commands `train-opportunity-ranker` and `predict-opportunity-ranker`.
- [x] Add focused tests for opportunity labels/features, threshold behavior, no-trade/allow predictions, and CLI parsing.
- [x] Run targeted tests and full training / prediction comparison.
- [x] Promote the hybrid variant `v42_gate_v4_rank` through the generic `predict-model` layer.
- **Status:** complete

## Active Initiative: Project File Cleanup

### Goal

整理项目根目录、测试临时目录和报告产物，保留源码、配置、文档、最终报告、模型和可复用数据缓存；删除已由最终摘要覆盖的中间样本文件以及明确的缓存/临时测试输出。

### Cleanup Phases

- [x] Inventory current Git state, root files, largest artifacts, and generated directories.
- [x] Identify no-regret cleanup targets: `__pycache__`, `.pytest_cache`, `.pytest_tmp`, `.tmp_tests`, `tmp_pytest_run`, root `.tmp_*` diagnostics, and `src/a_share_analyzer.egg-info`.
- [x] Identify report intermediates with final summaries: large TradingView factor sample/detail CSVs and pattern backtest detail/trade/forward-price CSVs.
- [x] Delete safe temporary artifacts and selected intermediate CSVs.
- [x] Update `.gitignore` so future test scratch directories and `.tmp_*` files stay out of the worktree.
- [x] Verify resulting Git status and remaining large files.
- **Status:** complete

### Cleanup Decisions

| Decision | Rationale |
|----------|-----------|
| `runcmd` is no longer active | The code path and tests were deleted during model/project cleanup; any old `command.txt` usage is historical only. |
| Keep source/config/docs/model artifacts | Current worktree contains active feature changes and trained model outputs. |
| Keep final report JSON/summary files | They are the compact results used by later review and documentation. |
| Delete cache/test scratch directories | They are reproducible, noisy, and not project source. |
| Delete sample/detail CSVs only where summary outputs exist | This follows the user's instruction to remove middle results when a route already has final results. |
| Xueqiu profile is historical | The active archive code path has been removed; retained data/profile state is no longer part of the current execution flow. |
| Remove obsolete source modules after model cleanup | `runcmd`, Xueqiu archive, old probability workflow, old V3/V3.1/V4 wrappers, and their tests are no longer part of the active project. |

## Phases

## Historical Initiative: Mainboard 20d TP Probability Model

### Goal

Build a replacement for TradingView aggregate scoring that ranks A-share mainboard stocks by `success_prob`: the probability of hitting +10% take-profit within 20 trading days before a -8% stop-loss, using only data available at the signal date.

Status as of 2026-05-05: superseded by the V4.2/V4 hybrid `predict-model` flow and removed from active source.

### Implementation Phases

### Phase 1: Label and Configuration Foundation

- [x] Add path-aware `label_tp10_sl8_20d` generation using `t+1` open as entry price.
- [x] Mark success, stop-loss failure, timeout failure, and same-day take-profit/stop-loss conflict.
- [x] Exclude conflict samples from model training and evaluation.
- [x] Add probability config fields for `take_profit_return`, `stop_loss_return`, `entry_price_mode`, and conflict handling.
- [x] Preserve compatibility with existing `label_stable_up` tests or migrate tests intentionally.
- **Status:** complete

### Phase 2: Daily Feature Revision

- [x] Keep raw daily price, volume, amount, moving-average, volatility, drawdown, RSI, MACD, ADX, CCI, Williams %R, Stochastic, and Stochastic RSI features.
- [x] Add explicit recent-high proximity and long downtrend repair features as model inputs, not hard filters.
- [x] Remove TradingView aggregate score columns from probability feature selection.
- [x] Add tests ensuring `all_rating`, `ma_rating`, `osc_rating`, and `avg_all_rating_5d` do not enter training features.
- **Status:** complete

### Phase 3: Weekly and Monthly Features

- [x] Build weekly OHLCV/amount features aligned to each sample date without future full-week leakage.
- [x] Build monthly OHLCV/amount features aligned to each sample date without future full-month leakage.
- [x] Add weekly returns, MA distances, RSI, MACD, ADX, volume ratios, amount ratios, range position, and drawdown features.
- [x] Add monthly returns, MA distances, RSI, MACD, volume ratios, amount ratios, range position, and drawdown features.
- [x] Add tests that higher-timeframe features do not use future rows inside the same week.
- **Status:** complete

### Phase 4: Dataset and Model Training

- [x] Update `build_probability_dataset` to use the new path label and conflict exclusion.
- [x] Keep hard filters limited to mainboard scope, data quality, history length, and liquidity.
- [x] Keep XGBoost as the primary model; defer an interpretable baseline until model comparison is needed.
- [x] Persist the label column and feature columns with the model artifact.
- **Status:** complete

### Phase 5: Evaluation

- [x] Replace stable-up evaluation with take-profit/stop-loss evaluation.
- [x] Report Top 10 / Top 20 take-profit hit rate, stop-loss rate, timeout rate, average outcome days, average 20-day return, and lift versus baseline.
- [x] Keep ROC-AUC, PR-AUC, Brier score, and log loss as secondary diagnostics.
- [x] Add probability-bucket outputs for monotonicity checks.
- **Status:** complete

### Phase 6: Prediction Output and Reporting

- [x] Output `success_prob` ranking with key explanation fields.
- [x] Include daily, weekly, and monthly explanation fields in CSV and terminal summary.
- [x] Add `risk_notes` as descriptive hints only, not as filters.
- [x] Ensure prediction uses the same feature columns saved during training.
- **Status:** complete

### Phase 7: Verification and Documentation

- [x] Run targeted label, feature, dataset, model, evaluation, and probability workflow tests.
- [x] Run a smoke train/predict workflow through `tests/test_probability_workflow.py`.
- [x] Update README command descriptions and field explanations.
- [x] Record test results and residual risks in `progress.md`.
- **Status:** complete

### Active Decisions

| Decision | Rationale |
|----------|-----------|
| Mainboard only | User confirmed scope; avoids expanding universe complexity |
| Entry price is `t+1` open | Avoids future leakage and matches practical next-day execution |
| Success means +10% take-profit within 20 trading days | User wants high-probability 20-day upside selections |
| Stop-loss threshold is -8% | Models stable upside under a defined adverse path constraint |
| Timeout without +10% is failure | User selected strict objective |
| Same-day TP/SL conflict is excluded | Daily bars cannot identify intraday order |
| TradingView aggregate scores are forbidden features | Historical validation showed negative usefulness |
| TradingView component indicators may be used as raw indicators | RSI/MACD/ADX/etc. can still carry useful information |
| High-position/overheated and long-downtrend-unrepaired states are features, not filters | User explicitly rejected these as hard filters |
| Daily, weekly, and monthly volume/amount features are required | User requested all three timeframes include volume information |
| Train three path classes and sort by risk-adjusted score | Binary take-profit probability raised stop-loss risk; ranking now penalizes stop-loss probability |
| Use one horizon-conditioned model across 5/10/20/40 day targets | User wants a single model that jointly considers multiple horizons and target combinations |

### Phase 1: Requirements & Discovery

- [x] 明确用户目标为 A 股主板技术面分析，不涉及自动交易
- [x] 明确交付形式为命令行工具
- [x] 明确以日线为主、分钟线为辅
- [x] 明确三类模板：趋势突破、回调低吸、强势股跟踪
- [x] 将设计整理为正式 spec
- **Status:** complete

### Phase 2: Planning & Structure

- [x] 固定项目目录结构
- [x] 定义数据源抽象接口
- [x] 定义股票池过滤规则
- [x] 定义基础指标清单与字段规范
- [x] 定义三类模板的配置结构
- [x] 定义 CLI 命令入口
- **Status:** complete

### Phase 3: Implementation

- [x] 初始化 Python 项目结构
- [x] 实现 AKShare 数据源适配器
- [x] 实现主板股票池模块
- [x] 实现日线更新和本地缓存
- [x] 实现基础指标计算
- [x] 实现三类策略模板
- [x] 实现 CLI 命令
- **Status:** complete

### Phase 4: Testing & Verification

- [x] 为股票池过滤编写测试
- [x] 为指标计算编写测试
- [x] 为模板筛选编写测试
- [x] 跑通端到端样例
- [x] 记录验证结果和已知限制
- **Status:** complete

### Phase 5: Delivery

- [ ] 整理最终使用说明
- [ ] 说明数据源限制与后续扩展点
- [ ] 交付可运行的 V1
- **Status:** in_progress

## Key Questions

1. 如何在不依赖具体第三方字段名的情况下统一股票列表、日线和分钟线接口？
2. 主板过滤规则在免费数据源下如何稳定表达并可测试？
3. 三类模板的参数结构如何设计，才能兼顾可读性和可扩展性？

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| V1 仅做分析，不做交易执行 | 保持问题边界清晰，避免过早引入回测和下单复杂度 |
| 命令行是第一阶段唯一交付形式 | 用户已明确要 CLI，最适合快速迭代 |
| 日线是主筛选层 | 免费数据源更适合日线主流程，分钟线放在增强层更稳 |
| 分钟线仅用于候选股二次确认 | 降低数据量和接口不稳定性对系统主干的影响 |
| V1 采用 AKShare 为原型主源 | 免费、起步快，适合先跑通流程 |
| 数据源必须抽象成统一接口 | 后续切换付费源时不重写上层逻辑 |
| 本地缓存优先使用 Parquet | 实现简单、读取快、适合表格型行情数据 |
| 三类策略采用模板加参数形式 | 避免每个想法都变成一个独立脚本 |
| `intraday-screening` 的盘中排序作为独立汇总层实现 | 保持原有日线输出链路不变，降低回归风险 |
| `daily_score` 直接复用 `watchlist._stable_score()` | 避免盘中链路引入第二套日线评分口径 |
| 最终盘中结果按 `intraday_5m_score` 排序 | 让本交易日 5 分钟状态成为首要排序依据 |
| `update` 直接改为自动补全末尾缺口 | 用户明确不再需要新命令，且不允许通过更晚的 `--start-date` 制造缺口 |
| `daily-screening` 的趋势复核通过独立 `trend` 指令接入 | 复用现有趋势评分链路，避免在 `pattern/watchlist` 内重复实现一套打分 |
| `watchlist` 采用“旧体系通过 AND trend 宽松阈值通过” | 保留原有技术候选稳定性，同时叠加 `buy_score / price_action_score` 质量复核 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `rg.exe` 在当前环境启动失败 | 1 | 改用 PowerShell 原生命令探索文件 |
| 当前目录不是 Git 仓库，无法提交 spec | 1 | 用户后续已初始化 Git，阻塞已解除 |
| `pip install -e .` 在沙箱内无法写入用户临时目录 | 1 | 升级权限后完成依赖安装 |
| `stock_zh_a_spot_em()` 在当前网络环境下不稳定 | 1 | 改为 `stock_info_a_code_name()` 做股票池更新 |
| `planning-with-files` 的 catchup 脚本路径在当前机器不存在 | 1 | 直接复用仓库内已有 planning files 继续记录实现过程 |

## Notes

- 实现阶段禁止把分钟线扩张为全市场主数据层
- 优先验证“能稳定筛出合理候选股”，而不是追求指标数量
- 若用户后续要求接入付费分钟线，应先保持当前接口不变再替换底层实现
- 趋势复核第一版只影响 `watchlist` 准入，不改 `watchlist` 现有排序逻辑
- `pattern` 体系已进入六模式阶段：新 `1/2/3` 为 `量顶天立地` 三阶段，旧 `2/3/4` 已顺延为新 `4/5/6`

## V4.2 Opportunity-Gated Ranker Plan

- [x] 写入 V4.2 设计规格
- [x] 实现机会日标签、机会门模型和条件排序模型
- [x] 接入 `train-opportunity-ranker` / `predict-opportunity-ranker` CLI
- [x] 增加单测和 CLI 回归测试
- [x] 跑完整 V4.2 训练与 V4 基线对比
- [x] 记录结论：机会门有价值，个股排序器仍未泛化
- [x] 增加 `v42_gate_v4_rank` 对照：机会门 + V4 原排序
- [x] 重训并验证 hybrid 优于 V4.2 ranker
- **Status:** complete
