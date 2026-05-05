# Findings & Decisions

## Requirements

- 面向 A 股主板做技术面分析
- 不做自动交易、下单、回测撮合
- 需要日线，最好能补充分钟线
- 目标是筛出满足条件的股票，不是做交易系统
- 交付形式为命令行工具
- 关注周期为短线和波段
- 策略框架需覆盖趋势突破、回调低吸、强势股跟踪三类模板
- 数据源先免费跑通，后续保留切换付费源能力

## Research Findings

- 当前项目目录起步状态接近空仓库，仅有设计文档，无现成代码基础。
- AKShare 适合作为 A 股免费原型数据源，适合先跑通股票列表和日线流程。
- Tushare Pro 更适合后续升级为稳定或更完整的数据源，尤其在分钟线场景下更有长期可维护性，但不适合作为当前 V1 的第一依赖。
- V1 的主要技术风险不在策略表达，而在股票池定义、数据源稳定性和字段标准化。
- 对当前需求而言，日线主筛选加分钟线辅助确认是最合适的复杂度控制方式。
- 在当前网络环境下，AKShare 的东财全市场实时分页接口容易出现代理或连接中断。
- `stock_info_a_code_name()` 在当前环境下可成功返回股票代码和名称，更适合用于 V1 的股票池更新。
- `baostock` 在当前环境下可成功登录、拉取全市场证券列表，并成功抓取主板日线与 5 分钟线样例。
- 四个 `pattern` 当前最稳妥的共同约束落点是在 `evaluate_strategies()` 公共入口，而不是导出结果时做后置过滤。
- “最近 200 日内任意 5 日涨幅达到或超过 10%” 更适合实现为独立公共配置，而不是挂在某一个 `type` 策略配置下。
- 现有 `type1`、`type2`、`type3` 测试样本长度不足以直接承载默认 `200` 日历史门槛，因此原有形态测试需要显式缩小测试口径，避免和新门槛耦合。
- `intraday-screening` 当前虽然名为盘中复筛，但原实现只会刷新候选股日线并生成三份日线结果，不会产出真正的 5 分钟排序层。
- 现有 `watchlist._stable_score()` 已经是项目内唯一稳定使用的日线质量分公式，直接复用它最能避免口径漂移。
- `patterns_all_*.csv` 可能对同一股票输出多条记录，因此盘中排序层需要先聚合 `pattern_ids`，再为 `daily_score` 选择一个主 `pattern_id`。
- `tradingview_avg5_*.csv` 与 `macd_divergence_*.csv` 的 symbol 列仍保留 Excel 友好格式 `=\"000001\"`，盘中汇总层必须先做统一归一化。
- 5 分钟量价背离与 MACD 背离的比较方向不同：量价背离关注“新低/新高但量缩”，MACD 背离关注“新低/新高但 DIF 反向”。两者不能共用同一比较器。
- 当前 `update` 原实现会无条件按 `--start-date -> --end-date` 全量重拉并覆盖 parquet，确实没有利用本地缓存末尾日期。
- 用户最终放弃新增 `append` 命令，要求直接把 `update` 改成自动补全模式。
- `update` 的新口径是：本地不存在文件时 `--start-date` 生效；本地已存在文件时，总是从本地最后日期的下一天开始补，用户传更晚的 `--start-date` 也不会制造缺口。
- 阈值研究链路已实现为独立命令 `research-thresholds`，不会修改当前默认交易阈值，只负责输出历史样本、分布对比、候选阈值和阈值回测对比。
- 阈值研究当前采用“先全量评分样本、再按 `daily/weekly/monthly` 抽样交易日、再按 `next_open` 回测”的实现路径，避免把阈值门槛提前混入样本构建。
- 强势组当前按未来收益前 `20%` 定义，弱势组按未来收益 `< 0` 定义，底部组按未来收益后 `20%` 定义。
- 候选阈值当前按 `loose / balanced / strict` 三档生成，来源于 `all_p50`、`weak_p60/weak_p80`、`strong_p20` 等分位数，而不是人工写死。
- 组合阈值研究当前输出 `candidate_buy_only`、`candidate_buy_plus_*` 与 `current_default_rules` 对比，便于直接比较“研究候选阈值”和“当前默认阈值”。
- 当前默认执行链路已支持按 `signal_type` 分开阈值：`breakout` 使用研究得到的新门槛，`pullback` 继续走全局默认门槛。
- 当前启用的 `breakout` 默认覆盖值为 `buy_score >= 81.3308` 且 `price_action_score >= 75.0373`；`pullback` 暂未启用独立覆盖。
- `daily-screening` 现有的全局动量初筛和 `type1~type4` 涨幅门槛仍然是第一层准入，新增趋势复核更适合做旧 `watchlist` 之后的一道严格交集过滤。
- 复用 `scan_indicator_scored_entries(...)` 作为 `trend` 指令底层实现最稳妥，可以直接拿到 `buy_score`、`price_action_score` 和现有 MACD/量价字段，而不用在 `daily-screening` 里复制评分逻辑。
- `watchlist` 过滤层最适合直接复用旧 `watchlist` 的已有排序结果，只按 `symbol` 做严格交集和阈值过滤，再把趋势字段补写回候选项。
- `MACD` 金叉死叉、MACD 背离、量价背离更适合先以离散状态字段输出到 `trend` 和 `watchlist`，而不是第一版就纳入硬筛选。
- 新一轮 `pattern` 重构已把 `量顶天立地` 抽为共享母形态，并拆成 `预突破 / 突破确认 / 突破后延续或回踩` 三阶段。
- 新 `pattern1/2/3` 使用共享老前高与突破事件检测；旧 `pattern2/3/4` 顺延为新 `pattern4/5/6`。
- 老前高当前实现为“最近的合格局部峰值”，要求左右各 `10` 日窗口中的峰值、距今至少 `60` 日、后续至少 `10%` 回撤、且后续阶段曾有 `low < MA60`。
- 突破事件当前实现为“阳线 + 最高价突破前高 + 当日成交量 >= 前 20 日均量 3 倍”，其中 20 日均量不含突破日。
- `pattern3` 当前采用“突破后 1-8 日内、收盘不超过前高 10% 上方；若盘中跌破 MA20，则收盘必须站回 MA20”的硬条件。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 项目采用分层结构：数据源、股票池、指标、筛选、策略、输出 | 降低耦合，便于替换数据源和增量扩展 |
| 数据接口统一为 `get_instruments` / `get_daily_bars` / `get_intraday_bars` | 锁定上层依赖边界 |
| V1 不引入数据库 | 文件缓存足够，先优化开发速度 |
| V1 基础指标只保留 MA、新高、均量、涨跌幅、回撤、波动率 | 用最小指标集覆盖三类模板 |
| 模板逻辑不硬编码为独立脚本 | 统一配置和筛选引擎更利于维护 |
| 根目录提供 `main.py` 作为直接入口 | 用户无需先安装包，也能先跑 CLI |
| 股票池更新改用 `stock_info_a_code_name()` | 比 `stock_zh_a_spot_em()` 更轻，当前网络下更稳定 |
| 默认 provider 切换为 `baostock` | 当前环境下 `baostock` 实测比 AKShare-东财链路更稳定 |
| 历史连续上涨门槛作为 `history_momentum_filter` 独立配置块接入 `AppConfig` | 这是四个 pattern 的共同准入门槛，不属于某一个 type 的局部规则 |
| 在 `evaluate_strategies()` 前统一检查历史短窗涨幅 | 能保证四个 pattern 自动共享同一规则，且不遗漏其他调用路径 |
| 5 分钟盘中排序新增独立模块 `intraday_ranking.py` | 让盘中评分逻辑脱离 CLI，便于单测和后续调权 |
| `intraday_5m_score` 以 50 为中性基准，并对各事件做加减分 | 比纯规则标签更适合用于排序，又保留了分项解释性 |
| `intraday-screening` 新增 `intraday_rank_<date>.csv` | 把现有日线结果与 5 分钟事件放到同一张可排序表里 |
| `update` 的增量策略留在 CLI/存储层，不下推到 provider | Provider 继续只负责按给定区间取数，上层决定是否增量 |
| `trend` 指令作为全市场趋势复核输出单独落到 `reports/trend/` | 让 `daily-screening`、人工分析和后续研究都能复用同一份趋势评分结果 |
| `watchlist_trend_filter` 作为独立配置块挂到 `AppConfig` | 把 `watchlist` 复核阈值和 `trend_entry_rules` 的交易阈值分开，避免语义混淆 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 当前目录不是 Git 仓库，无法完成 skill 要求中的 commit | 记录为外部前置条件，不阻塞规划与文档写入 |
| `rg` 工具在当前环境拒绝访问 | 改用 PowerShell 文件探索命令 |
| `pip install -e .` 在默认沙箱中无法写用户临时目录 | 通过权限提升安装依赖 |
| `stock_zh_a_spot_em()` 真实调用失败 | 用更轻的代码列表接口替换股票池更新来源 |
| AKShare 日线接口在当前环境下持续出现代理中断 | 改为接入并默认启用 `baostock` |
| `planning-with-files` 的 session catchup 路径不存在 | 放弃脚本恢复，直接在仓库现有 planning files 上继续记录 |

## Resources

- 设计文档: `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-04-09-a-share-analysis-design.md`
- 20 日止盈概率设计文档: `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-05-01-mainboard-20d-tp-probability-design.md`
- AKShare 股票数据文档: https://akshare.akfamily.xyz/data/stock/stock.html
- AKShare 数据说明: https://akshare.akfamily.xyz/data_tips.html
- Tushare 分钟数据说明: https://tushare.pro/document/1?doc_id=234
- 项目入口: `C:\Users\wdyab\Desktop\wdy\stocks\main.py`

## Session: 2026-05-01 Mainboard 20d TP Probability

### Requirements

- 新方法用于替代 TradingView 聚合分数，目标是得到未来 20 日上涨概率高的主板股票选法。
- 股票池仅限 A 股主板。
- 标签采用路径规则：`t+1` 开盘价入场，20 个交易日内先触发 +10% 止盈为成功。
- 先触发 -8% 止损为失败；20 日内没有触发 +10% 也算失败。
- 同一日同时触发 +10% 止盈与 -8% 止损时，日线无法判断先后，样本剔除。
- TradingView 聚合分数不参与训练和排序。
- TradingView 组成指标的原始指标值可以作为普通技术指标候选特征。
- “当前价格距离近期高点过近且过热”和“长期下降趋势未修复”不能作为硬过滤，只能作为模型特征。
- 日线、周线、月线都需要加入成交量和成交额特征。
- 月线需要加入 RSI。
- 每日输出需要包含关键解释字段，帮助人工复核概率排序。

### Codebase Findings

- 当时的概率链路包括：`labels.py`、`features.py`、`ml_dataset.py`、`ml_models.py`、`ml_evaluation.py`、`probability_reporting.py` 和 `train-prob / predict-prob` CLI。
- 当时 `labels.py` 的 `add_forward_labels` 是收盘收益 + 最大回撤标签，不是路径感知止盈止损标签。
- 当时 `ProbabilityConfig` 只有 `horizon_days`、`min_future_return`、`max_future_drawdown`、`min_history_days`、`top_n_list`，需要补充止盈止损语义字段。
- 当时 `features.py` 会调用 `add_technical_ratings`，且 `numeric_feature_columns` 会把数值型聚合评分列纳入训练特征，必须显式排除。
- 当时 `ml_models.py` 只支持 XGBoost，符合主模型方向，但还没有解释性基线模型。
- 当时 `ml_evaluation.py` 的 Top N 评估只统计 hit rate、未来收益和回撤，需要扩展到止盈率、止损率、timeout、outcome days 和 lift。
- 当时 `probability_reporting.py` 的预测摘要还会展示 `all_rating`，需要移除并改为关键解释字段。
- 当时 `tests/test_probability_workflow.py` 已经覆盖 train/predict 流程，可作为新标签和新特征改造后的回归入口。

Note as of 2026-05-05: this probability workflow has been superseded and removed from active source. The current project no longer has `labels.py`, `ml_dataset.py`, `ml_models.py`, `ml_evaluation.py`, `probability_reporting.py`, `train-prob`, or `predict-prob`.

### Implementation Findings

- `add_take_profit_stop_loss_labels` now implements the path-aware +10%/-8%/20-day label while preserving the legacy `add_forward_labels` API.
- `build_probability_dataset` now uses the new configured label column and excludes same-day TP/SL conflict samples.
- `numeric_feature_columns` now explicitly excludes TradingView aggregate scores and path-label target fields.
- Weekly and monthly feature generation uses period bars aligned by the period's last available trade date, so a daily row does not receive a later week/month aggregate.
- Prediction reports now remove `all_rating` from the probability summary and include daily/weekly/monthly explanation fields plus descriptive `risk_notes`.
- The implementation keeps XGBoost as the primary model. Logistic regression baseline support is deferred to a later model-comparison pass.
- The probability dataset now expands each stock-date into configured horizon targets, allowing one model to learn 5/10/20/40 day path behavior through `horizon_days`, `take_profit_return`, and `stop_loss_return`.
- Prediction now scores every configured horizon per stock and aggregates per-horizon risk-adjusted scores into `ensemble_score`.

## Visual/Browser Findings

- AKShare 官方文档显示其股票数据能力足以支撑 A 股原型阶段的日线分析流程。
- AKShare 文档同时提示部分数据接口存在字段或复权层面的使用注意事项，因此上层不能直接耦合其原始字段。
- Tushare 官方文档显示分钟数据属于更明确的专业数据能力，适合在后续升级阶段接入，而不是 V1 的主依赖。
- 官方文档显示 `stock_zh_a_hist_min_em` 的 1 分钟数据只返回近 5 个交易日且不复权，因此分钟线应保持辅助定位。

## Session: 2026-05-02 Project File Cleanup

### Inventory Findings

- Worktree is already dirty with active source, config, README, planning, probability-report, and new TradingView factor research changes; cleanup must avoid reverting those.
- `.gitignore` ignores common Python artifacts, `data/`, CSV/parquet outputs, `.env.local`, and `command.txt`, but it does not yet ignore `.tmp_tests/`, `.pytest_tmp/`, `tmp_pytest_run/`, or root `.tmp_*` diagnostics.
- `command.txt` / `runcmd` was previously treated as a command-entry convenience, but the active `runcmd` code path has now been removed.
- Root `.tmp_baostock_login_stack.txt`, `.tmp_is_trading_day_stack.txt`, `.tmp_rdns_false.txt`, and `.tmp_rdns_true.txt` are diagnostic scratch files and safe cleanup candidates.
- `__pycache__`, `.pytest_cache`, `.pytest_tmp`, `.tmp_tests`, `tmp_pytest_run`, and `src/a_share_analyzer.egg-info` are reproducible generated artifacts.
- `reports/tradingview_factor/` has compact final JSON and summary CSVs plus a 603 MB `tradingview_factor_samples_2024-01-01_2026-04-30.csv`; the sample CSV is an intermediate behind the final research summary.
- `reports/backtests/patterns/` has final summary JSON/CSV and stop-grid summary/best CSVs, plus large detail/forward/trade CSVs that are intermediate analysis inputs already reflected by summaries.
- `data/xueqiu/1155695148/browser_profile` was previously retained for possible archive refreshes, but the active Xueqiu archive code path has now been removed.
- After cleanup, the largest remaining files are model artifacts and retained data/report outputs rather than large research CSV intermediates.
- `.pytest_tmp`, `tmp_pytest_run`, and `.tmp_tests/pytest-temp/pytest-of-wdyab` remain because both normal and escalated deletion attempts returned Windows `Access denied`.

## Session: 2026-05-05 V4.2 Opportunity-Gated Ranker

### Requirements

- Keep V4 risk filtering as a hard first-stage gate.
- Add a date-level opportunity model so the system can choose no-trade days.
- Optimize for higher Top20/Top50 20-day average return, higher buy-day win rate, and balanced risk.
- Allow lower trading coverage; target useful coverage is at least 30% of evaluable days.
- Avoid label leakage: date-level labels may use future returns as targets, but the TopN set used for labels must be selected from OOF or historical-available scores.

### Baseline Findings

- V4 risk filter still works: validation/test risk AUC about 0.706/0.704.
- V4 test Top20 baseline: average 20-day return 0.779%, win rate 26.88%, stop-loss rate 5.16%.
- V4.1 did not solve ranking: test candidate-pool Spearman about -0.009 and Top20 outcomes worse than V4.

### Implementation Findings

- V4.2 is now implemented as: V4 risk hard gate -> date-level opportunity gate -> conditional stock ranker.
- The opportunity gate is trained on daily aggregated candidate-pool conditions and selects a conservative threshold on validation Top20 outcomes.
- The conditional ranker is trained only on risk-passed samples from opportunity-allowed days, using long-quality style 20d/60d return targets.
- The selected threshold is intentionally strict: validation coverage is 51/128 days, test coverage is 34/128 days.
- The gate improves trade discipline but does not yet solve stock selection. On test, Top50 improved average 20d return and win rate versus V4 baseline, but Top20 deteriorated and the stock-level rank correlation is still negative.
- Current best interpretation: keep the date-level opportunity gate as a promising component, but do not promote the V4.2 conditional ranker as the main buy selector without another iteration.
- Follow-up hybrid test supports this interpretation: using the opportunity gate with V4 `long_upside_score` ranking outperforms both the V4 all-day baseline and the V4.2 conditional ranker on test Top20/Top50 average return and take-profit rate.
- Hybrid trade-off: coverage drops to 31/128 test days, and Top20 bad-risk rate is 15.00%, slightly worse than V4 baseline. The next risk work should tune the opportunity gate/risk gate jointly rather than train another stock ranker first.
- Daily screening should use the hybrid through the generic `predict-model` layer, not by adding a separate daily-screening stage. This preserves the existing daily sequence and lets watchlist consume the model through its existing `reports/predict_model/predictions_<date>.csv` contract.

## Session: 2026-05-05 Project Code Cleanup and Current Architecture

### Current Architecture

- The active daily flow is now `daily-screening -> update -> tradingview -> predict-model -> macd -> atr -> trend-universe -> trend -> pattern -> watchlist`.
- `predict-model` is the stable integration name for the current model. It writes `reports/predict_model/predictions_YYYY-MM-DD.csv`.
- The current model version is `v42_gate_v4_rank`: V4.2 opportunity gate decides whether the day is tradable, and V4 `long_upside_score` ranks the low-risk candidate pool.
- `watchlist_pattern` treats model output as a hard gate: `trade_permission = allow` and `action = candidate` are required before a pattern candidate can enter the watchlist.
- `final_score_v42` and `buy_score_v42` are now the model ranking fields used by watchlist sorting. TradingView aggregate scores remain as technical context, not the primary ranking layer.

### Removed / Superseded Code Paths

- Deleted old probability workflow modules and tests: `labels.py`, `ml_dataset.py`, `ml_models.py`, `ml_evaluation.py`, `probability_reporting.py`, and the related probability tests.
- Deleted old utility/archive modules and tests: `runcmd.py`, `tools/runcmd.py`, `plotting.py`, `pattern_scan.py`, `xueqiu_archive.py`, `xueqiu_rendering.py`.
- Removed obsolete V3/V3.1/V4 wrapper CLI paths from the active command surface, including old buy-trigger, clean-win, alpha-ranker, risk-gated, long-quality, and stacked-value entry points.
- The remaining model commands are `train-opportunity-ranker`, `predict-opportunity-ranker`, and `predict-model`.

### Verification Findings

- Core compile check passed for `cli.py`, `config.py`, `models.py`, `paths.py`, `watchlist.py`, and `stacked_trade_value.py`.
- Targeted regression passed: `tests/test_cli.py`, `tests/test_watchlist.py`, and `tests/test_stacked_trade_value.py` reported `62 passed`.
- Full test run reached `181 passed`; the remaining `2 errors` were Windows pytest temporary-directory `PermissionError` failures, not business assertion failures.
