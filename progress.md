# Progress Log

## Session: 2026-05-08 Mainline and Picks Refresh

### Status

- **Status:** complete

### Actions Taken

- Loaded `planning-with-files` and checked the local catchup helper; native Codex session parsing is not implemented, so no external context was imported.
- Read current `主线.md`, `选股.md`, `docs/picks-writing-guide.md`, `reports/xueqiu/1155695148.md`, `reports/xueqiu/其他博主.md`, and latest `watchlist_2026-05-07.json`.
- Read `track_stock.xlsx` Sheet2 with the bundled runtime; the only tracked row is `002245`.
- Recorded that the existing `2026.5.7` section in `选股.md` must be replaced because the risk-score field uses the old raw-risk style and the user asked to ignore old same-date picks.
- Updated `主线.md` to mark local Xueqiu data cutoff as `2026-05-07`, keep the first-tier `半导体 + AI硬件/算力` conclusion, and add a stricter stock-picking note that mainline names still need P1/P2, trend, and ATR checks.
- Replaced the `2026.5.7` section in `选股.md` with a fresh list based only on latest `watchlist_2026-05-07.json`: `002491`, `002929`, `000889`, `000818`, `603938`, and `002995`.
- Added the required tracking-stock summary for `002245`, marked as `不适合` because P1/P2 are `19.04 / 7.32`.

### Verification

- Confirmed `选股.md` contains exactly one `### 2026.5.7` heading.
- Confirmed old same-date raw risk values and old low-P1/P2 main-table rows were removed from the new `2026.5.7` section.
- Confirmed the final-picks, trading-aid, and tracking-stock tables have consistent column counts.

## Session: 2026-05-06 Daily Watchlist Semantics

### Status

- **Status:** implemented; targeted watchlist tests passing

### Actions Taken

- Changed `trade_permission` from a watchlist entry hard gate to a next-open trading-environment warning.
- Changed main daily `watchlist` construction to combine:
  - low-risk model Top20 names from `predict-model`
  - low-risk pattern matches from `patterns_all`
- Kept `watchlist_pattern` as the low-risk pattern subset.
- Added watchlist top-level fields:
  - `trade_permission`
  - `next_open_trade_permission`
  - `next_open_trade_warning`
  - `trade_permission_note`
- Updated `docs/picks-writing-guide.md` so daily writing starts with next-open permission and treats `no_trade` days as observation-only.

### Verification

- `python -m py_compile src\stocks_analyzer\watchlist.py src\stocks_analyzer\cli.py`
- `python -m pytest tests\test_watchlist.py -q`
- `python -m pytest tests\test_cli.py::test_build_parser_accepts_current_model_commands -q`
- `python -m pytest tests\test_daily_screening.py -q`

## Session: 2026-05-05 Lightweight Walk-Forward Validation

### Status

- **Status:** implemented; targeted verification passed
- **Purpose:** verify whether the current mainline `v42_gate_v4_rank` result generalizes across multiple future time windows instead of relying on one train/valid/test split.

### Actions Taken

- Added design spec: `docs/superpowers/specs/2026-05-05-lightweight-walkforward-validation-design.md`.
- Added `validate-model-walkforward` CLI command.
- Added chronological window generation over actual trading dates.
- Added in-memory V4.2 hybrid retraining for each window without overwriting the daily `predict-model` artifact.
- Added per-window TopN metrics and aggregate summary reports.
- Added Top20 diagnostic threshold flags:
  - mean win rate >= `0.70`
  - worst-window win rate >= `0.55`
  - mean 20d return > `0.05`
  - mean stop-loss rate <= `0.06`
  - mean bad-risk rate <= `0.15`
  - mean coverage >= `0.20`

### Verification

- `python -m py_compile src\stocks_analyzer\stacked_trade_value.py src\stocks_analyzer\cli.py`
- `python -m pytest tests\test_cli.py::test_build_parser_accepts_current_model_commands tests\test_stacked_trade_value.py::test_generate_walkforward_windows_are_chronological tests\test_stacked_trade_value.py::test_generate_walkforward_windows_returns_empty_when_history_is_short tests\test_stacked_trade_value.py::test_summarize_walkforward_topn_metrics_adds_threshold_flags -q` -> 4 passed

### Reports

- `reports/model_walkforward/v42_walkforward_windows.csv`
- `reports/model_walkforward/v42_walkforward_topn_metrics.csv`
- `reports/model_walkforward/v42_walkforward_summary.csv`
- `reports/model_walkforward/v42_walkforward_config.json`

### Next Step

Run a real multi-window job, preferably overnight:

```bash
python -m stocks_analyzer --project-root . validate-model-walkforward --model v42 --windows 8 --max-iter 40 --top-n 20,50
```

## Session: 2026-05-05 V5.1 Candidate-Only Ranker Implementation

### Status

- **Status:** complete, experimental; not promoted to `predict-model`
- **Reason:** V5.1 improved candidate-pool rank diagnostics, but Top20/Top50 trading metrics did not beat V4.2 or V5.

### Actions Taken

- Implemented V5.1 candidate-only rank labels, daily candidate rank percentiles, and ordinal rank grades.
- Added candidate-pool cross-sectional rank features.
- Added V5.1 ranker training with optional LightGBM LambdaRank and histogram boosting fallback.
- Added validation-selected blend with V4.2 ranking score.
- Added V5.1 training/prediction artifact save/load, reports, prediction table, and CLI commands:
  - `train-candidate-ranker`
  - `predict-candidate-ranker`
  - aliases: `train-v51-candidate-ranker`, `predict-v51-candidate-ranker`
- Kept `predict-model` unchanged on V4.2 hybrid.

### Verification

- `python -m py_compile src\stocks_analyzer\stacked_trade_value.py src\stocks_analyzer\cli.py`
- `python -m pytest tests\test_stacked_trade_value.py -q` -> 20 passed
- `python -m pytest tests\test_cli.py::test_build_parser_accepts_current_model_commands tests\test_stacked_trade_value.py::test_v51_candidate_rank_labels_are_daily_candidate_only tests\test_stacked_trade_value.py::test_v51_candidate_ranker_scores_and_blends_candidates -q` -> 3 passed

### Training Run

- Command: `python -m stocks_analyzer --project-root . train-candidate-ranker --max-iter 40 --top-n 20,50 --predict-date 2026-04-30`
- Model artifact: `data/ml/v51_candidate_ranker/v51_candidate_ranker.pkl`
- Metadata: `data/ml/v51_candidate_ranker/v51_candidate_ranker_metadata.json`
- Reports: `reports/v51_candidate_ranker/`
- Prediction output: `reports/v51_candidate_ranker/predictions_2026-04-30.csv`
- Engine: `lightgbm_lambdarank`
- Feature count: `82`
- Selected blend: `0.65 * V5.1 ranker + 0.35 * V4.2 rank score`

### Key Metrics

| Split | TopN | Model | Avg 20d Return | Win Rate | TP20 | SL20 | Bad Risk |
|---|---:|---|---:|---:|---:|---:|---:|
| valid | 20 | V4.2 hybrid | 0.1077 | 0.8627 | 0.4443 | 0.0076 | 0.0437 |
| valid | 20 | V5 | 0.1047 | 0.8563 | 0.4342 | 0.0120 | 0.0449 |
| valid | 20 | V5.1 | 0.1034 | 0.8563 | 0.4272 | 0.0114 | 0.0449 |
| test | 20 | V4.2 hybrid | 0.1193 | 0.8000 | 0.4786 | 0.0286 | 0.1182 |
| test | 20 | V5 | 0.1247 | 0.8048 | 0.4976 | 0.0238 | 0.1068 |
| test | 20 | V5.1 | 0.1224 | 0.7833 | 0.5024 | 0.0357 | 0.1409 |

### Ranker Diagnostics

| Split | Candidate Rows | Spearman | NDCG@20 |
|---|---:|---:|---:|
| train | 34,719 | 0.2851 | 0.8908 |
| valid | 45,042 | 0.2441 | 0.6639 |
| test | 12,328 | 0.2810 | 0.6304 |

### Conclusion

V5.1 learned a measurable candidate-pool ranking signal, but the selected Top20 degraded on validation and test. The ranker is useful diagnostically, but not suitable for promotion. The next direction should not be another ranker wrapper; it should investigate why rank diagnostics do not translate into top-pick trading metrics, especially label shape, top-heavy loss, and interaction with no-trade/opportunity days.

## Session: 2026-05-05 V5 Volume-Price Fusion Implementation

### Status

- **Status:** complete, experimental; not promoted to daily `predict-model`
- **Reason:** V5 test Top20 improved slightly, but validation metrics did not improve consistently versus `v42_gate_v4_rank`.

### Actions Taken

- Implemented V5 multi-period daily volume-price features in `stacked_trade_value.py`.
- Added 1-day trigger/risk hints, 5-day buy-point confirmation fields, 20-day volume-price structure fields, and 5-day versus 20-day acceleration/divergence fields.
- Added sparse rule-based `volume_price_extreme_risk_flag`.
- Added volume-price risk label and healthy-path quality target.
- Added volume-price risk classifier, quality regressor, OOF scoring helper, V5 learned fusion model, prediction report, artifact save/load, and comparison metrics.
- Added CLI commands:
  - `train-volume-price-fusion`
  - `predict-volume-price-fusion`
- Kept `predict-model` on current V4.2 hybrid path because V5 is not yet a clear upgrade.
- Added focused V5 tests for feature coverage, extreme-risk flag behavior, label quality, submodel scoring/fusion scoring, and CLI parsing.

### Verification

- `python -m py_compile src\stocks_analyzer\stacked_trade_value.py src\stocks_analyzer\cli.py`
- `python -m pytest tests\test_stacked_trade_value.py -q` -> 18 passed
- `python -m pytest tests\test_cli.py::test_build_parser_accepts_current_model_commands -q` -> 1 passed
- `python -m pytest tests\test_cli.py::test_build_parser_accepts_current_model_commands tests\test_stacked_trade_value.py::test_v5_volume_price_features_cover_1d_5d_20d_windows tests\test_stacked_trade_value.py::test_v5_extreme_volume_price_risk_flags_high_volume_weak_candle tests\test_stacked_trade_value.py::test_v5_submodels_and_fusion_score_generate_rankable_fields -q` -> 4 passed

### Training Run

- Command: `python -m stocks_analyzer --project-root . train-volume-price-fusion --reuse-base-artifact --max-iter 40 --top-n 20,50 --predict-date 2026-04-30`
- Model artifact: `data/ml/v5_volume_price_fusion/v5_volume_price_fusion.pkl`
- Metadata: `data/ml/v5_volume_price_fusion/v5_volume_price_fusion_metadata.json`
- Reports: `reports/v5_volume_price_fusion/`
- Prediction output: `reports/v5_volume_price_fusion/predictions_2026-04-30.csv`
- Split dates: train end `2025-03-06`, validation end `2025-09-25`, test end `2026-04-30`
- Base mode: `reuse_existing_v42_artifact`

### Key Metrics

| Split | TopN | Model | Avg 20d Return | Win Rate | TP20 | SL20 | Bad Risk |
|---|---:|---|---:|---:|---:|---:|---:|
| valid | 20 | v42_gate_v4_rank | 0.1077 | 0.8627 | 0.4443 | 0.0076 | 0.0437 |
| valid | 20 | v5_volume_price_fusion | 0.1047 | 0.8563 | 0.4342 | 0.0120 | 0.0449 |
| test | 20 | v42_gate_v4_rank | 0.1193 | 0.8000 | 0.4786 | 0.0286 | 0.1182 |
| test | 20 | v5_volume_price_fusion | 0.1247 | 0.8048 | 0.4976 | 0.0238 | 0.1068 |
| valid | 50 | v42_gate_v4_rank | 0.0942 | 0.8111 | 0.3777 | 0.0187 | 0.0559 |
| valid | 50 | v5_volume_price_fusion | 0.0932 | 0.8076 | 0.3643 | 0.0203 | 0.0585 |
| test | 50 | v42_gate_v4_rank | 0.1112 | 0.7448 | 0.4524 | 0.0343 | 0.1391 |
| test | 50 | v5_volume_price_fusion | 0.1075 | 0.7314 | 0.4333 | 0.0324 | 0.1373 |

### Residual Risk

- The rigorous `retrain_v42_hybrid_oof` base path is implemented, but the full `max_iter=80` run exceeded one hour on the current machine.
- The successful full run used `--reuse-base-artifact`; this is useful for quick V5 evaluation, but the next serious comparison should either optimize the retrain-base path or run it overnight.
- The volume-price quality model is still weak in the candidate pool: validation candidate Spearman correlation was about `0.028`, test candidate Spearman correlation was about `-0.113`.

## Session: 2026-04-09

### Phase 1: Requirements & Discovery

- **Status:** complete
- **Started:** 2026-04-09
- Actions taken:
  - 读取并遵循 `brainstorming` 与 `analyze-research-question` 两个 skill
  - 探查当前项目目录，确认基本为空且不是 Git 仓库
  - 与用户逐步确认市场范围、分析类型、分析周期、策略方向、数据源策略和交付形式
  - 形成 A 股主板技术面分析框架的设计结论
  - 将设计写入正式 spec 文档
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-04-09-a-share-analysis-design.md` (created)

### Phase 2: Planning & Structure

- **Status:** complete
- Actions taken:
  - 读取 `planning-with-files` skill 和模板
  - 将已确认的需求、技术决策、阻塞条件写入持久化 planning files
  - 将下一阶段收束为目录结构、接口定义、模板结构和 CLI 设计
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\task_plan.md` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\findings.md` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\progress.md` (created)

### Phase 3: Implementation

- **Status:** complete
- Actions taken:
  - 初始化 `pyproject.toml`、默认配置、`src/stocks_analyzer` 包结构和测试目录
  - 实现 AKShare 数据源抽象、主板股票池构建、日线缓存、指标计算、三类模板筛选和结果输出
  - 增加根目录 `main.py`，支持未安装时直接运行 CLI
  - 在真实环境下完成 `update-universe` 和 `update-daily --limit 3` 烟雾验证
  - 跑通 `screen --all --as-of 2026-04-09` 与 `report --date 2026-04-09`
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\pyproject.toml` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\main.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\...` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\...` (created)

### Phase 4: Testing & Verification

- **Status:** complete
- Actions taken:
  - 使用 `python -m compileall src tests` 做静态语法检查
  - 修复测试导入路径问题，新增 `tests/conftest.py`
  - 运行 `python -m pytest -q`，当前 3 个单测全部通过
  - 验证空结果场景会落地 signals/report 文件，避免后续 `report` 命令报缺文件
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\conftest.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\cli.py` (updated)

### Phase 5: Delivery

- **Status:** in_progress
- Actions taken:
  - 根据真实报错补强 `update-daily` 的鲁棒性
  - 为日线抓取增加单只股票重试、失败跳过和批量总结
  - 新增 `--skip-existing`，支持断点续跑
  - 本地验证 `python main.py update-daily --start-date 20240101 --limit 3 --skip-existing` 可正常跳过已有缓存
  - 新增 `BaoStockDataProvider`，并将默认 provider 切换为 `baostock`
  - 实测 `python main.py update-universe` 与 `python main.py update-daily --start-date 20240101 --limit 3` 已可经由 `baostock` 成功执行
  - 为四个 `pattern` 新增共享的历史连续上涨门槛配置 `history_momentum_filter`
  - 在 `evaluate_strategies()` 入口加入最近窗口动量历史检查，统一过滤掉过去一段时间内从未出现过短窗强涨的股票
  - 补充 3 个策略回归测试，覆盖“允许通过”、“最近窗口内排除”和“更早历史达标但最近窗口不达标”的边界行为
  - 运行 `pytest tests/test_cli.py tests/test_daily_screening.py tests/test_strategies.py -q`，24 个测试全部通过
  - 基于确认过的 spec 新增 `src/stocks_analyzer/intraday_ranking.py`，实现盘中 5 分钟事件汇总、`intraday_5m_score` 计算和排序 CSV 输出
  - 在 `_run_intraday_screening(...)` 中接入新的盘中排序步骤，新增 `intraday_rank_<date>.csv` 与 `intraday_rank_path`
  - 复用 `watchlist._stable_score()` 生成 `daily_score`，避免引入第二套日线评分公式
  - 为量价背离、MACD 背离、金叉死叉、均线事件、排序与失败降级补充针对性测试
  - 运行 `pytest tests/test_intraday_ranking.py tests/test_intraday_screening.py -q`，7 个测试通过
  - 运行 `pytest tests/test_cli.py tests/test_intraday_screening.py tests/test_intraday_ranking.py -q`，20 个测试通过
  - 新增 `docs/superpowers/specs/2026-04-14-update-auto-append-design.md`，明确 `update` 改为末尾自动补全而不是新增 `append` 指令
  - 将 `update` 的单股逻辑抽成 `_update_daily_cache_for_symbol(...)`，实现“首次初始化 / 已覆盖跳过 / 从末尾下一天补缺口 / 合并去重写回”
  - 补充 `update` 回归测试，覆盖首次初始化、自动起点优先、已覆盖跳过与合并去重
  - 运行 `pytest tests/test_update_fallback.py tests/test_cli.py -q`，18 个测试通过
  - 运行 `pytest tests/test_daily_screening.py tests/test_intraday_screening.py tests/test_update_fallback.py -q`，7 个测试通过
  - 新增 `src/stocks_analyzer/trend_threshold_research.py`，实现阈值研究样本构建、分布统计、候选阈值生成、单指标阈值评估和组合阈值评估
  - 在 CLI 中新增 `research-thresholds` 命令，并接入独立的 `reports/threshold_research/` 报表输出
  - 补充 `tests/test_trend_threshold_research.py` 与 CLI 解析回归，覆盖抽样、候选阈值、单指标评估和组合评估
  - 运行 `pytest tests/test_trend_threshold_research.py -q`，4 个测试通过
  - 运行 `pytest tests/test_cli.py -q`，26 个测试通过
  - 运行 `pytest tests/test_trend_trading.py tests/test_trend_threshold_research.py tests/test_intraday_screening.py tests/test_update_fallback.py tests/test_watchlist.py tests/test_pattern_tradingview_scores.py -q`，25 个测试通过
  - 使用最小真实数据工作区运行 `research-thresholds --date 2025-10-20 --start-date 2025-09-01 --sample-mode weekly`，成功生成样本、分布、候选阈值和组合阈值回测文件
  - 将 `trend_entry_rules` 扩展为“全局默认 + signal_type 覆盖”配置结构
  - 在 `trend_indicator_scores.select_tradable_entries()` 中接入分信号阈值过滤，保持无 `signal_type` 数据仍走全局默认
  - 默认启用 `breakout` 研究阈值：`buy_score >= 81.3308`、`price_action_score >= 75.0373`；`pullback` 继续沿用全局默认阈值
  - 补充配置加载和分信号过滤测试，运行 `pytest tests/test_trend_trading.py tests/test_trend_threshold_research.py -q`，18 个测试通过
  - 使用最小真实数据工作区运行 `trend-entries --date 2025-09-12`，确认分信号阈值链路可正常输出结果
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\data_sources\akshare_provider.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\cli.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\storage.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\README.md` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\data_sources\baostock_provider.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\models.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\config.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\strategies.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_strategies.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\intraday_ranking.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_intraday_ranking.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_intraday_screening.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-04-14-update-auto-append-design.md` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_update_fallback.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\trend_threshold_research.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\trend_reporting.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_trend_threshold_research.py` (created)

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| spec 自检 | 检查占位符和结构一致性 | 无占位符、无明显矛盾 | 通过 | pass |
| 项目状态检查 | `git status --short --branch` | 若为 Git 仓库则返回状态 | 当前目录不是 Git 仓库 | known_limit |
| 语法检查 | `python -m compileall src tests` | 全部源码可编译 | 通过 | pass |
| 单元测试 | `python -m pytest -q` | 测试通过 | 3 passed | pass |
| CLI 帮助 | `python main.py --help` | 显示命令帮助 | 通过 | pass |
| 股票池更新 | `python main.py update-universe` | 写入主板股票池 | 写入 3062 条 | pass |
| 日线更新样例 | `python main.py update-daily --start-date 20240101 --limit 3` | 缓存日线文件 | 成功缓存 3 只股票 | pass |
| 筛选链路样例 | `python main.py screen --all --as-of 2026-04-09` | 输出或保存结果 | 当前样例无候选，已保存空结果 | pass |
| 报告读取样例 | `python main.py report --date 2026-04-09` | 读取已保存结果 | 成功输出“无候选” | pass |
| 断点续跑样例 | `python main.py update-daily --start-date 20240101 --limit 3 --skip-existing` | 已缓存标的被跳过 | 成功跳过 3 只 | pass |
| BaoStock 股票池 | `python main.py update-universe` | 刷新股票池 | 成功写入 3272 条 | pass |
| BaoStock 日线样例 | `python main.py update-daily --start-date 20240101 --limit 3` | 缓存 3 只样例日线 | 成功缓存 3 只 | pass |
| 共享历史动量过滤回归 | `pytest tests/test_cli.py tests/test_daily_screening.py tests/test_strategies.py -q` | 配置加载、选股入口和新过滤逻辑均正常 | 24 passed | pass |
| 盘中评分单测 | `pytest tests/test_intraday_ranking.py tests/test_intraday_screening.py -q` | 盘中事件、排序和新报告路径正常 | 7 passed | pass |
| 盘中接入回归 | `pytest tests/test_cli.py tests/test_intraday_screening.py tests/test_intraday_ranking.py -q` | CLI 接口与盘中新增逻辑兼容 | 20 passed | pass |
| update 增量回归 | `pytest tests/test_update_fallback.py tests/test_cli.py -q` | 首次初始化、增量起点、跳过与去重正常 | 18 passed | pass |
| update 链路兼容回归 | `pytest tests/test_daily_screening.py tests/test_intraday_screening.py tests/test_update_fallback.py -q` | update 改动未破坏现有筛选链路 | 7 passed | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-09 | `rg.exe` 启动失败，访问被拒绝 | 1 | 改用 PowerShell 文件命令 |
| 2026-04-09 | `git status` 返回非 Git 仓库 | 1 | 记录为环境前置条件，不阻塞规划 |
| 2026-04-09 | `pip install -e .` 在沙箱内权限不足 | 1 | 升级权限后安装依赖 |
| 2026-04-09 | `stock_zh_a_spot_em()` 网络请求失败 | 1 | 改用 `stock_info_a_code_name()` 更新股票池 |
| 2026-04-09 | `report` 与 `screen` 并行验证时抢先读取文件 | 1 | 改为顺序验证，并保留空结果文件输出 |
| 2026-04-09 | `update-daily` 因单只股票连接中断而整批退出 | 1 | 增加单只重试、失败跳过和 `--skip-existing` |
| 2026-04-09 | AKShare 东财日线接口持续被代理中断 | 1 | 切换到 BaoStock provider 并实测通过 |
| 2026-04-14 | 盘中排序汇总在合并多份 CSV 时误删 `name` 列 | 1 | 单独汇总 `name` 字段后再删辅助列 |
| 2026-04-14 | 量价背离与 MACD 背离共用比较器导致方向错误 | 1 | 拆分为独立的 volume divergence 比较函数 |
| 2026-04-14 | `update` 的新需求从新增 `append` 反复切换为直接改 `update` | 1 | 将最终口径固化到 spec，并只实现“末尾下一天自动补全” |
| 2026-04-18 | 组合回归 `pytest tests/test_cli.py tests/test_watchlist.py tests/test_daily_screening.py tests/test_trend_trading.py tests/test_intraday_screening.py tests/test_update_fallback.py -q` 在当前时限内未跑完 | 1 | 改为逐文件验证新增点和相邻链路，确认 parser、watchlist、daily-screening、trend_trading、intraday 和 update 子集均通过 |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 5: Delivery |
| Where am I going? | 整理交付说明，并根据用户反馈继续增强策略和数据链路 |
| What's the goal? | 构建 A 股主板技术面分析 CLI，先跑通股票池、日线、指标、模板和结果输出 |
| What have I learned? | 轻量代码列表接口比全市场实时接口更适合当前网络环境 |
| What have I done? | 已完成设计、实现、单测和最小真实链路验证 |

## Session: 2026-04-18

### Feature: Daily Screening Trend Filter

- **Status:** complete
- Actions taken:
  - 根据已确认 spec 新增 `trend` CLI 命令，输出 `reports/trend/trend_YYYY-MM-DD.csv/json`
  - 在趋势评分结果中新增 `macd_cross_state`、`macd_divergence_state`、`volume_price_divergence_state`
  - 扩展配置模型，新增 `watchlist_trend_filter.enabled / buy_score_min / price_action_score_min`
  - 在 `daily-screening` 中新增可选 `trend` 阶段；启用配置时按严格交集重写最终 `watchlist`
  - 在 `watchlist` 侧实现趋势记录存在性检查、`buy_score / price_action_score` 宽松阈值过滤，以及趋势字段回填
  - 更新 README 说明新的 `trend` 命令和 `watchlist` 复核语义
  - 补充测试覆盖 CLI parser、`watchlist` 趋势过滤、`daily-screening` 趋势阶段以及趋势评分新状态字段
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\models.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\config.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\trend_indicator_scores.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\trend_reporting.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\watchlist.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\daily_screening.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\cli.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_cli.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_watchlist.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_daily_screening.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_trend_trading.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\README.md` (updated)

## Test Results (2026-04-18)

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| CLI parser 子集 | `pytest tests/test_cli.py -k "trend or daily_screening or intraday_screening" -q` | 新 `trend` 命令和相关筛选命令解析正常 | 7 passed | pass |
| watchlist 趋势过滤 | `pytest tests/test_watchlist.py -q` | 严格交集与宽松阈值过滤正确 | 4 passed | pass |
| daily-screening 趋势阶段 | `pytest tests/test_daily_screening.py -q` | 启用 `watchlist_trend_filter` 时增加 `trend` 阶段并重写 watchlist | 2 passed | pass |
| 趋势评分回归 | `pytest tests/test_trend_trading.py -q` | 新状态字段不破坏既有趋势评分链路 | 13 passed | pass |
| 盘中链路回归 | `pytest tests/test_intraday_screening.py -q` | 新配置模型未破坏盘中链路 | 1 passed | pass |
| update 回归 | `pytest tests/test_update_fallback.py -q` | 新配置模型未破坏 update 链路 | 5 passed | pass |

## Session: 2026-04-21

### Feature: Volume Top Breakout Pattern Redesign

- **Status:** complete
- Actions taken:
  - 新增 `docs/superpowers/specs/2026-04-21-volume-top-breakout-design.md`，固化 `量顶天立地` 母形态与新 `pattern1/2/3`
  - 扩展配置模型为 `type1~type6`，并将旧 `pattern2/3/4` 顺延为新 `pattern4/5/6`
  - 新增 `src/stocks_analyzer/volume_top_breakout.py`，统一实现老前高选择与首次有效突破事件识别
  - 重写 `src/stocks_analyzer/strategies.py`，将新 `pattern1/2/3` 改为共享检测器驱动，并保留旧平台突破、趋势回踩、二波逻辑为 `pattern4/5/6`
  - 更新 `cli.py` 的模式映射、标签映射、`pattern` 命令帮助与导出列顺序
  - 更新 `watchlist.py` 的 pattern 优先级与分层规则，使其兼容 `1~6` 编号
  - 更新 `README.md` 与 `四个模式.md`，同步六模式口径
  - 重写并扩展 `tests/test_strategies.py`，覆盖新 `pattern1/2/3` 与顺延后的 `4/5/6`
  - 调整 `tests/test_cli.py`、`tests/test_intraday_ranking.py` 以适配新策略名与 pattern 优先级
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-04-21-volume-top-breakout-design.md` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\volume_top_breakout.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\models.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\config.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\strategies.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\cli.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\reporting.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\watchlist.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_strategies.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_cli.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_intraday_ranking.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\README.md` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\四个模式.md` (updated)

## Test Results (2026-04-21)

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 语法检查 | `python -m compileall src tests` | 新增六模式与共享检测器源码均可编译 | 通过 | pass |
| 策略回归 | `pytest tests/test_strategies.py -q` | 新 `pattern1/2/3` 与顺延后 `4/5/6` 均可命中 | 9 passed | pass |
| watchlist / 盘中排序回归 | `pytest tests/test_watchlist.py tests/test_intraday_ranking.py -q` | 新优先级与多模式聚合正常 | 17 passed | pass |
| 日常筛选链路回归 | `pytest tests/test_daily_screening.py tests/test_intraday_screening.py -q` | 新模式映射未破坏 daily / intraday screening | 3 passed | pass |
| CLI 相关子集回归 | `pytest tests/test_cli.py -k 'pattern or watchlist or daily_screening or intraday_screening or build_parser_accepts_pattern_flags or trend_summary or atr_summary or macd_summary or trend_universe_summary' -q` | 新 `pattern` 映射、导出和 watchlist 更新正常 | 8 passed | pass |
| 组合子集回归 | `pytest tests/test_strategies.py tests/test_watchlist.py tests/test_intraday_ranking.py tests/test_daily_screening.py tests/test_intraday_screening.py tests/test_cli.py -k 'pattern or watchlist or daily_screening or intraday_screening or build_parser_accepts_pattern_flags or trend_summary or atr_summary or macd_summary or trend_universe_summary' -q` | 相关链路组合运行正常 | 23 passed, 38 deselected | pass |

## Session: 2026-05-01

### Feature: Mainboard 20d Take-Profit Probability Model

- **Status:** implemented
- Actions taken:
  - Confirmed user objective: mainboard-only probability ranking for stocks likely to hit +10% within 20 trading days under a -8% stop-loss path rule.
  - Confirmed same-day take-profit/stop-loss conflict samples should be excluded.
  - Confirmed TradingView aggregate scores must not participate, while raw component indicators can remain candidate features.
  - Confirmed high-position/overheated and long-downtrend-unrepaired states should be model features, not hard filters.
  - Confirmed daily, weekly, and monthly volume/amount features are required, and monthly RSI is required.
  - Wrote and committed design spec `docs/superpowers/specs/2026-05-01-mainboard-20d-tp-probability-design.md` at commit `b0a87fa`.
  - Added active implementation phases to `task_plan.md`.
  - Recorded codebase findings in `findings.md`.
  - Implemented path-aware `label_tp10_sl8_20d` with `t+1` open entry, +10% take-profit, -8% stop-loss, timeout failure, and same-day conflict exclusion.
  - Added probability config fields for take-profit, stop-loss, entry mode, conflict handling, and label column.
  - Added daily high-position/downtrend-repair features and weekly/monthly OHLCV/amount/indicator features.
  - Excluded TradingView aggregate scores and path-label target fields from model features.
  - Updated dataset building, evaluation summaries, prediction reports, and model metadata for the new label.
  - Updated README probability command semantics.
- Files created/modified:
  - `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-05-01-mainboard-20d-tp-probability-design.md` (created and committed)
  - `C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\labels.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\features.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\ml_dataset.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\ml_evaluation.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\ml_models.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\probability_reporting.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\models.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\config.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\src\stocks_analyzer\cli.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_labels.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_features.py` (created)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_ml_dataset.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\tests\test_probability_workflow.py` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\README.md` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\task_plan.md` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\findings.md` (updated)
  - `C:\Users\wdyab\Desktop\wdy\stocks\progress.md` (updated)

## Test Results (2026-05-01)

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Targeted probability suite | `pytest tests/test_labels.py tests/test_features.py tests/test_ml_dataset.py tests/test_ml_evaluation.py tests/test_ml_models.py tests/test_probability_workflow.py tests/test_cli.py -k "prob or train_prob or predict_prob or build_parser" -q` | Label, features, dataset, model, evaluation, workflow, and CLI probability paths pass | 26 passed, 33 deselected; one XGBoost device warning | pass |
| TradingView compatibility | `pytest tests/test_tradingview_command.py tests/test_pattern_tradingview_scores.py tests/test_technical_ratings.py -q` | Existing TradingView command and scoring tests still pass | 6 passed | pass |
| Compile check | `python -m compileall src tests` | Source and tests compile | pass | pass |

### Residual Notes

- XGBoost emitted a device mismatch warning on this machine when CUDA is available but input data is on CPU. It is a performance warning, not a correctness failure.
- Logistic regression baseline support was deferred; the current trained probability model remains XGBoost.

## Session: 2026-05-01 Risk-Adjusted Path Model

### Feature: Three-Class Take-Profit / Stop-Loss / Timeout Model

- **Status:** implemented
- Actions taken:
  - Added `outcome_class` alongside the existing binary take-profit label.
  - Changed default probability training label to `outcome_class`.
  - Updated XGBoost training to use multiclass probabilities when the label has three classes.
  - Added prediction columns `take_profit_prob`, `stop_loss_prob`, `timeout_prob`, `risk_adjusted_score`, and `expected_score`.
  - Changed prediction sorting and TopN evaluation to use `risk_adjusted_score` when available.
  - Updated evaluation to compute actual take-profit, stop-loss, and timeout rates from path outcomes.
  - Updated README to describe the three-class path model and risk-adjusted ranking.

## Test Results (2026-05-01, Risk-Adjusted Path Model)

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Core multiclass suite | `pytest tests/test_labels.py tests/test_ml_models.py tests/test_ml_evaluation.py tests/test_ml_dataset.py tests/test_probability_workflow.py -q` | Labels, multiclass model, evaluation, dataset, and workflow pass | 16 passed; one XGBoost device warning | pass |
| CLI probability workflow | `pytest tests/test_features.py tests/test_probability_workflow.py tests/test_cli.py -k "prob or train_prob or predict_prob or build_parser" -q` | Feature generation and train/predict CLI paths pass | 25 passed, 20 deselected; one XGBoost device warning | pass |

## Session: 2026-05-01 Horizon-Conditioned Ensemble Model

### Feature: Single Model Across 5/10/20/40 Day Targets

- **Status:** implemented
- Actions taken:
  - Added `probability.horizon_targets` config with default 5d `+5%/-5%`, 10d `+7%/-6%`, 20d `+10%/-8%`, and 40d `+15%/-10%`.
  - Expanded probability datasets from one row per stock-date to one row per stock-date-horizon.
  - Added `horizon_days`, `take_profit_return`, and `stop_loss_return` as model features.
  - Added `horizon_weight` for training sample weighting and prediction ensemble weighting.
  - Normalized future outcome fields so evaluation can compare mixed horizons.
  - Updated prediction to score all configured horizons per stock and aggregate them into `ensemble_score`.
  - Updated prediction CSV to include per-horizon probability and score columns such as `take_profit_prob_5d` and `risk_adjusted_score_20d`.
  - Updated README to describe the horizon-conditioned model.

## Test Results (2026-05-01, Horizon-Conditioned Ensemble Model)

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Horizon workflow suite | `pytest tests/test_ml_dataset.py tests/test_probability_workflow.py tests/test_ml_models.py tests/test_ml_evaluation.py -q` | Dataset expansion, workflow, model, and evaluation pass | 11 passed; one XGBoost device warning | pass |
| CLI/features suite | `pytest tests/test_labels.py tests/test_features.py tests/test_cli.py -k "prob or train_prob or predict_prob or build_parser" -q` | Labels, features, and probability CLI parser pass | 24 passed, 26 deselected | pass |
| Compile check | `python -m compileall src tests` | Source and tests compile | pass | pass |

## Session: 2026-05-02 Project File Cleanup

### Discovery

- Checked session catchup; native Codex parsing is not implemented, so there was no previous cleanup context to import.
- Checked `git status --short`; repository already has many active modified/untracked files, so cleanup is scoped to generated artifacts and selected report intermediates only.
- Inspected root files, `.gitignore`, planning files, largest artifacts, report directories, and references to candidate files.
- Historical decision: `command.txt` was kept during the first cleanup because it belonged to the old `runcmd` entrypoint. The active `runcmd` code has since been removed.
- Decision: remove reproducible caches/test scratch directories and selected large intermediate CSVs whose compact JSON/summary outputs are already present.

### Cleanup Actions

- Updated `.gitignore` to ignore `.pytest_tmp/`, `.tmp_tests/`, `tmp_pytest_run/`, and `.tmp_*`.
- Removed root diagnostic `.tmp_*` files, Python/pytest caches, `src/a_share_analyzer.egg-info`, selected TradingView factor sample/detail CSVs, selected pattern backtest detail/forward/trade CSVs, entry backtest detail CSV, and threshold sample CSV.
- Removed 498 accessible children under `.tmp_tests`; only the ACL-blocked `pytest-temp/pytest-of-wdyab` path remains.
- Removed Xueqiu browser cache directories/files while keeping likely session state such as `Local State`, `Default/Local Storage`, `Default/Network`, and `Sessions`.
- Normal and escalated deletion attempts both failed for `.pytest_tmp`, `tmp_pytest_run`, and `.tmp_tests/pytest-temp/pytest-of-wdyab` with Windows `Access denied`.

### Verification

- Confirmed selected intermediate files and `src/a_share_analyzer.egg-info` no longer exist.
- Confirmed remaining `.tmp_tests` contains only the ACL-blocked `pytest-temp` directory.
- Largest remaining files are retained model artifacts, retained Xueqiu profile state, and normal report CSV outputs.
- Final `git status --porcelain` summary: `895` tracked deletions, `27` modified files, and `13` untracked paths. Most deletions are previously tracked `.tmp_tests` artifacts plus root diagnostic `.tmp_*` files.
- Tests were not run after cleanup because doing so would recreate the same cache and temporary directories.

## Session: 2026-05-05 V4.2 Opportunity-Gated Ranker

### Start

- User approved V4.2 direction and asked to start implementation.
- Wrote and committed the V4.2 design spec at `docs/superpowers/specs/2026-05-05-v42-opportunity-gated-ranker-design.md`.
- Added V4.2 working plan and findings to planning files.
- Implemented V4.2 core functions in `stacked_trade_value.py`: opportunity-day aggregation, gate model, conditional ranker, threshold selection, evaluation, prediction reporting, artifact save/load.
- Added CLI parser and run handlers for `train-opportunity-ranker` and `predict-opportunity-ranker`.
- Syntax check passed for `src/stocks_analyzer/stacked_trade_value.py` and `src/stocks_analyzer/cli.py`.

### Completion

- Added focused V4.2 tests for opportunity-day labels, day-level no-trade blocking, CLI parser coverage, and train/predict workflow.
- Verification passed:
  - `python -m pytest tests\test_cli.py -q`: 43 passed.
  - `python -m pytest tests\test_stacked_trade_value.py -q`: 28 passed.
  - `python -m py_compile src\stocks_analyzer\stacked_trade_value.py src\stocks_analyzer\cli.py tests\test_stacked_trade_value.py tests\test_cli.py`: passed.
- Full V4.2 training completed with `train-opportunity-ranker --max-iter 80 --top-n 20,50 --predict-date 2026-04-30`.
- Artifacts written:
  - `data/ml/v42_opportunity_ranker/v42_opportunity_ranker.pkl`
  - `data/ml/v42_opportunity_ranker/v42_opportunity_ranker_metadata.json`
  - `reports/v42_opportunity_ranker/v42_topn_metrics.csv`
  - `reports/v42_opportunity_ranker/v42_opportunity_metrics.csv`
  - `reports/v42_opportunity_ranker/v42_ranker_metrics.csv`
  - `reports/v42_opportunity_ranker/v42_comparison.csv`
  - `reports/v42_opportunity_ranker/predictions_2026-04-30.csv`
- Main result:
  - V4.2 opportunity gate has usable signal, but the conditional stock ranker still fails to generalize.
  - Test Top20 under V4.2: avg 20d return 0.14%, win rate 23.68%, stop-loss rate 6.91%, bad-risk rate 12.06%, coverage 34/128 days.
  - Test Top50 under V4.2: avg 20d return 1.08%, win rate 31.35%, stop-loss rate 6.41%, bad-risk rate 11.82%, coverage 34/128 days.
  - V4 baseline test Top50: avg 20d return 0.84%, win rate 27.33%, stop-loss rate 5.48%, bad-risk rate 13.19%.
  - Date-level test opportunity AUC is 0.5720; precision is 58.82%, recall is 32.26%.
  - Candidate ranker test Spearman is -0.1144, so the new stock-level ranker should not become the primary stock selector yet.
- Prediction for 2026-04-30:
  - `opportunity_score` 0.5645 is below selected threshold 0.9103.
  - Final daily decision is `no_trade`; ranked rows are diagnostics, not buy signals for that date.

### Hybrid Follow-Up: Opportunity Gate + V4 Rank

- User approved testing the next direction: keep V4.2 opportunity gate but replace the stock-level ranker with V4 `long_upside_score`.
- Implemented `score_v42_v4_rank_frame` and added `--rank-source v4` for `predict-opportunity-ranker`.
- V4.2 training now writes a third comparison group: `v42_gate_v4_rank`.
- Added and passed tests for the hybrid score path, comparison inclusion, workflow prediction, and CLI parser.
- Verification:
  - `python -m pytest tests\test_cli.py -q`: 43 passed.
  - `python -m pytest tests\test_stacked_trade_value.py -q`: 30 passed.
  - `python -m py_compile src\stocks_analyzer\stacked_trade_value.py src\stocks_analyzer\cli.py`: passed.
- Full retraining completed with `train-opportunity-ranker --max-iter 80 --top-n 20,50 --predict-date 2026-04-30`.
- Hybrid selected opportunity threshold:
  - `opportunity_threshold`: 0.9333753680987827
  - validation coverage: 39/128 days
  - selected on validation Top20 objective
- Test comparison:
  - V4 baseline Top20: avg 20d return 0.779%, win rate 26.88%, TP 6.41%, SL 5.16%, bad-risk 14.45%, coverage 128 days.
  - V4.2 ranker Top20: avg 20d return 0.137%, win rate 23.68%, TP 5.29%, SL 6.91%, bad-risk 12.06%, coverage 34 days.
  - Hybrid Top20: avg 20d return 1.502%, win rate 32.90%, TP 11.13%, SL 6.61%, bad-risk 15.00%, coverage 31 days.
  - V4 baseline Top50: avg 20d return 0.839%, win rate 27.33%, TP 6.47%, SL 5.48%, bad-risk 13.19%, coverage 128 days.
  - V4.2 ranker Top50: avg 20d return 1.077%, win rate 31.35%, TP 6.76%, SL 6.41%, bad-risk 11.82%, coverage 34 days.
  - Hybrid Top50: avg 20d return 1.658%, win rate 31.81%, TP 10.26%, SL 5.81%, bad-risk 13.48%, coverage 31 days.
- Generated hybrid prediction:
  - `reports/v42_opportunity_ranker/predictions_v4_rank_2026-04-30.csv`
  - 2026-04-30 remains `no_trade` because `opportunity_score` 0.5645 is below hybrid threshold 0.9334.

### Daily Screening Predict Model Replacement

- Replaced the generic `predict-model` command with the current best model: V4.2 opportunity gate + V4 `long_upside_score` ranking.
- `predict-model` now saves `model_version=v42_gate_v4_rank` to `reports/predict_model/predictions_<date>.csv`.
- Old V3.1 buy-trigger compatibility was removed during the subsequent project cleanup; `predict-model` is now the active daily model integration point.
- Updated watchlist model-join logic:
  - V4.2 hybrid predictions use `trade_permission == allow` and `action == candidate` as hard model gates.
  - V4.2 hybrid watchlist sorting uses `final_score_v42` then `buy_score_v42`.
  - Old V3.1 prediction fields are no longer accepted by the active watchlist model join.
- Generated `reports/predict_model/predictions_2026-04-30.csv` with `v42_gate_v4_rank`; that date remains `no_trade`.
- Verification:
  - `python -m py_compile src\stocks_analyzer\cli.py src\stocks_analyzer\watchlist.py src\stocks_analyzer\predict_model.py`: passed.
  - `python -m pytest tests\test_cli.py -q`: 43 passed.
  - `python -m pytest tests\test_watchlist.py -q`: 10 passed.
  - `tests\test_daily_screening.py` could not run in this environment because pytest temp directory cleanup fails with Windows `PermissionError`; the failure happens during pytest temp setup/cleanup, not in a daily-screening assertion.

### Project Code Cleanup

- Removed obsolete strong-candidate and medium-candidate code paths after promoting `v42_gate_v4_rank` through `predict-model`.
- Deleted old modules:
  - `src/stocks_analyzer/pattern_scan.py`
  - `src/stocks_analyzer/runcmd.py`
  - `src/stocks_analyzer/plotting.py`
  - `src/stocks_analyzer/xueqiu_archive.py`
  - `src/stocks_analyzer/xueqiu_rendering.py`
  - `tools/runcmd.py`
- Deleted old probability workflow:
  - `src/stocks_analyzer/labels.py`
  - `src/stocks_analyzer/ml_dataset.py`
  - `src/stocks_analyzer/ml_evaluation.py`
  - `src/stocks_analyzer/ml_models.py`
  - `src/stocks_analyzer/probability_reporting.py`
- Removed old model CLI paths and public wrappers for stacked-value, risk-gated, clean-win, alpha-ranker, risk-upside, long-quality, and buy-trigger variants.
- Kept current model commands:
  - `train-opportunity-ranker`
  - `predict-opportunity-ranker`
  - `predict-model`
- Updated watchlist model join to require V4.2/V4 hybrid fields: `trade_permission`, `action`, `risk_score`, `long_upside_score`, `opportunity_rank_score`, `final_score_v42`, and `buy_score_v42`.
- Verification:
  - `python -m py_compile src\stocks_analyzer\cli.py src\stocks_analyzer\config.py src\stocks_analyzer\models.py src\stocks_analyzer\paths.py src\stocks_analyzer\watchlist.py src\stocks_analyzer\stacked_trade_value.py`: passed.
  - `python -m pytest tests\test_cli.py tests\test_watchlist.py tests\test_stacked_trade_value.py -q`: 62 passed.
  - Full `python -m pytest tests -q --basetemp C:\tmp\pytest-stocks-cleanup`: 181 passed, 2 Windows temp-directory `PermissionError` setup/cleanup errors.

### Documentation Status Sync

- Updated README to describe the current `daily-screening -> predict-model -> watchlist` architecture.
- Updated `task_plan.md` to mark V4.2 hybrid implementation complete and add the cleanup/documentation sync phase.
- Updated `findings.md` with the current active architecture and removed code paths.
- Updated this progress log with cleanup scope and verification results.
- Updated `docs/picks-writing-guide.md` so final pick writing treats model output as a hard filter and uses V4.2/V4 hybrid fields as soft ranking context.
