# A Share Analyzer

面向 A 股主板的命令行技术分析工具。

当前项目聚焦三件事：

- 更新主板股票池和本地日线缓存
- 生成 pattern、TradingView、MACD/量价状态、ATR 风险辅助等技术结果
- 基于技术规则生成 `watchlist`，并在次日盘中做小范围复筛和排序

项目不包含交易执行，也不会替你自动决定最终买入标的。

## 当前定位

当前流程里有两层结果：

- 技术结果层：`pattern`、`tradingview`、`macd`、`atr`
- 候选池层：`watchlist`、`watchlist_pattern`、`watchlist_trend`

其中：

- `daily-screening` 负责跑完整技术链路，并生成当日 `watchlist`、`watchlist_pattern`、`watchlist_trend`
- 主 `watchlist` 和 `watchlist_pattern` 会补入 ATR 辅助字段，并写入 `连续上榜天数`
- `intraday-screening` 负责读取上一交易日或指定日期的 `watchlist`，直接抓取候选股当日 5 分钟线做盘中复筛，并生成一份盘中排序 CSV
- `选股.md` 不在自动命令链里更新，留给你手动整理和最终决策

另外，项目现在新增了一条独立于 `pattern/watchlist` 的趋势交易研究链路：

- `trend`：输出指定交易日的 `breakout/pullback` 趋势评分结果，主要用于研究和展示
- `trend-universe`：定义并生成日 K 趋势股池，供 `pattern` 结果补充第一层趋势字段
- `trend-signals`：在趋势股池上识别 `breakout/pullback`
- `trend-score`：对 setup 叠加 MACD、RSI、BOLL、KDJ、ATR、量价等指标做收盘评分
- `trend-entries`：把收盘评分映射为“次日开盘”的买入候选
- `backtest-signals`：做 setup 层的固定持有回测
- `backtest-portfolio`：做 setup 层的组合回测
- `backtest-entries`：做“收盘评分，次日开盘买入”的单信号回测
- `backtest-entries-portfolio`：做“收盘评分，次日开盘买入”的组合回测
- `research-thresholds`：比较强弱组指标分布，生成阈值候选和阈值回测对比

这条链路的第一版用于研究统计优势，不做自动交易，也不预设止盈止损。

## 安装

环境要求：

- Python `>= 3.11`

推荐安装方式：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install playwright
```

安装完成后可以使用以下入口：

- `python -m stocks_analyzer`
- `mystock`
- `stocks-analyzer`

推荐统一写法：

```bash
python -m stocks_analyzer --project-root . --help
```

## 目录约定

常用目录和文件：

```text
config/                      配置文件
data/                        本地缓存数据
reports/patterns/            模式扫描结果
reports/tradingview/         TradingView 技术评分结果
reports/macd/                MACD/量价统一状态表
reports/atr/                 ATR 风险辅助表
reports/watchlists/          日终 watchlist
reports/daily_screening/     daily-screening 运行摘要
reports/intraday_screening/  盘中复筛结果
reports/trend_universe/      趋势股池明细与汇总
reports/trend_signals/       breakout/pullback setup 结果
reports/trend_scores/        收盘多指标评分结果
reports/trend_entries/       次日开盘买入候选
reports/backtests/signals/   setup 层固定持有回测
reports/backtests/portfolio/ setup 层组合回测
reports/backtests/entries/   次日开盘单信号回测
reports/backtests/entries_portfolio/ 次日开盘组合回测
reports/threshold_research/  阈值研究样本、分布、候选阈值和回测对比
选股.md                      你手动维护的选股记录
主线.md                      你手动维护的主线记录
```

## 快速开始

第一次初始化：

```bash
python -m stocks_analyzer --project-root . update --start-date 20240101
```

查看核心技术结果：

```bash
python -m stocks_analyzer --project-root . pattern --as-of 2026-04-10
python -m stocks_analyzer --project-root . tradingview --date 2026-04-10
python -m stocks_analyzer --project-root . macd --date 2026-04-10
python -m stocks_analyzer --project-root . atr --date 2026-04-10
```

执行一轮完整日终筛选并生成 `watchlist`：

```bash
python -m stocks_analyzer --project-root . daily-screening --date 2026-04-10
```

次日盘中只对 `watchlist` 做复筛：

```bash
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11
```

运行趋势交易研究链路：

```bash
python -m stocks_analyzer --project-root . trend-universe --date 2026-04-10
python -m stocks_analyzer --project-root . trend-signals --date 2026-04-10
python -m stocks_analyzer --project-root . trend-score --date 2026-04-10
python -m stocks_analyzer --project-root . trend-entries --date 2026-04-10
python -m stocks_analyzer --project-root . backtest-signals --date 2026-04-10 --start-date 2025-01-01
python -m stocks_analyzer --project-root . backtest-portfolio --date 2026-04-10 --start-date 2025-01-01
python -m stocks_analyzer --project-root . backtest-entries --date 2026-04-10 --start-date 2025-01-01
python -m stocks_analyzer --project-root . backtest-entries-portfolio --date 2026-04-10 --start-date 2025-01-01
python -m stocks_analyzer --project-root . research-thresholds --date 2026-04-10 --start-date 2025-01-01 --sample-mode monthly
```

## 命令说明

### `update`

作用：

- 不传股票代码时，刷新主板股票池并批量更新日线数据
- 传入股票代码时，只更新单只股票

当前更新行为：

- 首次初始化时，按 `--start-date -> --end-date` 拉取完整区间
- 本地已存在 `data/daily/<symbol>.parquet` 时，会自动检测最后一个交易日
- 本地已有缓存时，会从“本地最后日期的下一天”开始补齐后续缺口
- 如果本地数据已经覆盖 `--end-date`，会直接跳过，不再发请求

这意味着：

- `--start-date` 主要用于首次初始化
- 本地已有缓存时，`update` 会优先保证日期连续，但默认信任本地末尾数据，不允许通过更晚的 `--start-date` 制造缺口

当前推荐的数据源顺序：

- 日线更新优先建议走 `sina`
- 如果 `sina` 链路不稳定，再考虑切换或临时回退到 `baostock`
- 当前 `update` 命令内部会优先走 `akshare` 的日线实现，并默认把 `sina` 作为首选日线入口

常用示例：

```bash
python -m stocks_analyzer --project-root . update --start-date 20240101
python -m stocks_analyzer --project-root . update 603588 --start-date 20240101
python -m stocks_analyzer --project-root . update --start-date 20240101 --end-date 20260413 --limit 100
```

主要参数：

- `symbol`：可选，6 位股票代码
- `--start-date`：开始日期，格式 `YYYYMMDD`；首次初始化时生效
- `--end-date`：结束日期，格式 `YYYYMMDD`
- `--limit`：只更新前 N 只股票

### `pattern`

作用：

- 扫描本地日线缓存，识别 1 到 6 号模式

常用示例：

```bash
python -m stocks_analyzer --project-root . pattern
python -m stocks_analyzer --project-root . pattern --1
python -m stocks_analyzer --project-root . pattern --2 --6
python -m stocks_analyzer --project-root . pattern --as-of 2026-04-10 --output reports/my_patterns.csv
python -m stocks_analyzer --project-root . pattern --plot-all
```

主要参数：

- `--1 --2 --3 --4 --5 --6`：只识别指定模式
- `--as-of`：分析截止日期，格式 `YYYY-MM-DD`
- `--limit`：终端展示上限
- `--output`：自定义 CSV 输出路径
- `--plot-all`：为命中股票批量生成图形

默认输出：

- `reports/patterns/patterns_all_YYYY-MM-DD.csv`

#### 六个模式分别识别什么

##### 模式1：量顶天立地预突破型

这类股票通常在过去一段时间里出现过一个明显前高，且这个前高必须是前后各 40 个交易日内的局部最高点。之后经历了较长时间的回撤修复，当前仍未突破，但已经重新逼近老前高。未突破前高时不能放大量，检查日成交量相对 20 日均量不得超过 `1.5` 倍，避免前高下方放量滞涨。

它更适合拿来观察“第二天是否可能放量突破前高”。

##### 模式2：量顶天立地突破确认型

这类股票在完成底部修复后，已经出现过有效的量顶天立地突破：突破的老前高必须是前后各 40 个交易日内的局部最高点；突破日为阳线，最高价突破前高，成交量创近 90 个交易日新高，且突破日收盘位置、上影线和实体质量合格。当前检查日必须处在突破后 1 到 10 个交易日内，收盘价仍在前高上方；从突破日至今没有收盘有效跌破 `MA20 * 0.98`，且区间最高价相对突破日收盘价涨幅不超过 `10%`。

它更适合拿来观察“突破后是否继续站稳前高”的跟踪机会，不再把突破当天本身作为模式2信号。

##### 模式3：量顶天立地突破后延续/回踩型

这类股票已经在前几天完成了量顶天立地式突破，突破的老前高必须是前后各 40 个交易日内的局部最高点。目前仍处在突破后 1 到 10 个交易日内，当前收盘价回到前高下方，但仍守在 `MA20 * 0.98` 上方；从突破日至今所有收盘价都不能有效跌破这条 MA20 容忍线，区间最高价相对突破日收盘价涨幅也不能超过 `10%`。检查日还必须缩量，成交量低于 5 日均量。

它更适合拿来观察“突破后缩量回踩、二次上车”的候选机会。

##### 模式4：老鸭头鸭鼻孔金叉型

这类股票前面先走出一段“鸭颈”上涨，最近 `20-35` 个交易日前形成前后各 `20` 个交易日内的局部最高点作为“鸭头顶”。鸭头顶前 `30` 日内涨幅至少 `18%`，从鸭颈低点到鸭头顶涨幅至少 `20%`，鸭头顶本身要站在 `MA20/MA60` 上方。鸭头顶之后进入 `5-25` 日缩量回调，最低点相对鸭头顶至少回撤 `5%`，但回调低点不能有效跌破 `MA60` 的 `5%` 容忍线。

回调阶段要求整体成交量不超过鸭头顶附近三日均量的 `0.75`，后半段成交量不超过前半段的 `0.85`，单日最大量不超过鸭头顶附近三日均量的 `1.2`；回调中不能出现实体跌幅不低于 `4%`、成交量不低于回调均量 `1.5` 倍的放量大阴线。触发点是回调后 `MA5` 曾连续至少 `2` 天低于 `MA10`，且在回调最低点之后、最近 `8` 日内 `MA5` 再次上穿 `MA10`。金叉日成交量不超过 `20` 日均量的 `0.90`，金叉后到检查日 `MA5` 不能重新跌回 `MA10` 下方，最新收盘不能低于 `MA10` 的 `0.99` 倍。当前允许仍低于鸭头顶，不对距离鸭头顶下方多远作硬性要求，但不能已经高出鸭头顶超过 `3%`，否则不再视为鸭鼻孔低吸位置。

它更像是在找“老鸭头第一波上涨后的缩量洗盘末端，鸭嘴尚未完全张开前的试错低吸点”。

##### 模式5：趋势回踩型

这类股票近期刚打出过一个短期高点，随后在最近两天内缩量回踩到 `MA20` 附近甚至盘中短破，但收盘又重新站回 `MA20` 上方。趋势斜率必须同时成立：`MA20` 和 `MA60` 都要高于 1 日前，也要高于 10 日前；当前股价还必须在 `MA60` 上方。回踩缩量看的是触碰或收回 `MA20` 的那根 K 线，要求 `volume_ma_5 / volume_ma_20 <= 0.95`。

它更适合观察“强趋势里的深洗盘后，是否会重新转强”。

##### 模式6：拉升下降反抽型

这类股票前面经历了由一个放量阳开启的主升，之后快速下降并回踩到倍量阳收盘支撑线附近，若支撑不破或跌破后快速重新站回，就进入反抽观察。回踩阶段要求整体缩量、后半段继续缩量，回踩期平均量不超过锚点量的 `0.9`，从回踩低点到检查日的收盘价需要维持在支撑线 `±5%` 内，`support_hold` 分支最近 `10` 日内要触碰过支撑附近。另要求从峰值后一天开始的回踩阶段里，最大单日成交量不超过“峰值日及前 2 日”平均成交量的 `1.2` 倍，即 `pullback_max_rise_tail_volume_ratio <= 1.2`。

#### 纯模式回测结果（数据截止 2026-04-24）

以下结果来自 `backtest-patterns` 纯模式回测，统计口径为：`T` 日收盘命中模式，`T+1` 日开盘买入，固定持有 `5/10/20/40` 个交易日后按收盘价卖出。同一只股票、同一个模式在 5 个交易日内只统计第一次命中。该结果不叠加 `watchlist`、主线、TradingView、趋势池、MACD/量价等后续过滤条件。

注意：下表是 2026-04-24 当时规则下的历史快照。模式4已经由旧的平台突破改为 `duck_nostril_cross` 老鸭头鸭鼻孔金叉型，模式2/3/5/6 也有过规则调整，因此新版规则的统计表现需要重新运行 `backtest-patterns` 后再更新。

胜率评价口径：

- 单笔收益为正即计为胜出，公式为：`胜率 = 收益为正的样本数 / 总样本数`
- 收益计算使用 `T+1` 日开盘买入价和到期日收盘卖出价
- 胜率只回答“到期是否赚钱”，不回答“赚多少 / 亏多少”
- 因此胜率必须和 `平均收益`、`平均最大浮盈`、`平均最大回撤`、样本数一起看
- 当前所有模式裸信号胜率都低于 50%，说明模式本身更适合作为候选形态筛选器，不能单独当作最终买点

回测明细文件：

- `reports/backtests/patterns/pattern_backtest_details_2026-04-24.csv`
- `reports/backtests/patterns/pattern_backtest_summary_2026-04-24.csv`
- `reports/backtests/patterns/pattern_forward_prices_2026-04-24.csv`
- `reports/backtests/patterns/pattern_stop_grid_2026-04-24.csv`
- `reports/backtests/patterns/pattern_stop_grid_trades_2026-04-24.csv`

止盈止损网格研究口径：

- 数据来自 `pattern_forward_prices_2026-04-24.csv`，即抽样回测样本的 `T+1` 买入后 40 个交易日逐日价格
- 抽样口径：`2024-01-01` 到 `2026-04-24` 之间随机抽样 `80` 个交易日，`sample_seed=42`
- 固定止盈网格：`4% / 6% / 8% / 10% / 12% / 15% / 20% / 25% / 30%`
- 固定止损网格：`3% / 5% / 7% / 10% / 12% / 15%`
- 额外 MA20 止损：收盘价低于 `MA20 * 0.98` 时退出，即 `MA20` 下方 `2%` 容忍
- 同一天同时触发止盈和固定止损时按保守口径 `stop-first`
- 下表按“收益空间”排序，优先看 `avg_return_pct`，胜率只作为辅助指标

各模式当前最优止盈止损选择：

| 模式 | 策略名 | 持有周期 | 止盈 | 固定止损 | MA20止损 | 胜率 | 平均收益 | 止盈率 | 固定止损率 | MA20止损率 | 结论 |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `volume_top_pre_breakout` | 40日 | 30% | 5% | 收盘跌破 `MA20 * 0.98` | 39.52% | **2.81%** | 10.85% | 36.21% | 43.57% | 当前最强右尾收益模式 |
| 2 | `volume_top_breakout` | 40日 | 30% | 5% | 收盘跌破 `MA20 * 0.98` | 28.48% | **2.46%** | 17.58% | 58.79% | 22.42% | 弹性强，但失败率高，必须叠加强过滤 |
| 3 | `volume_top_follow_through` | 20日 | 15% | 10% | 收盘跌破 `MA20 * 0.98` | 34.81% | **0.75%** | 20.44% | 6.08% | 60.77% | 只适合作为补充观察 |
| 4 | `duck_nostril_cross` | 5日 | 4% | 3% | 收盘跌破 `MA20 * 0.98` | 44.81% | **0.17%** | 35.06% | 36.04% | 7.47% | 空间弱，更多是短线试错 |
| 5 | `trend_pullback` | 40日 | 25% | 7% | 收盘跌破 `MA20 * 0.98` | 26.90% | **0.69%** | 12.79% | 16.94% | 67.90% | 有少量右尾，但效率低 |
| 6 | `double_volume_support_rebound` | 5日 | 4% | 5% | 收盘跌破 `MA20 * 0.98` | 48.06% | **0.10%** | 24.50% | 10.85% | 61.09% | 不适合做收益空间策略 |

按收益空间优先的模式排序：

`模式1 > 模式2 > 模式3 > 模式5 > 模式4 > 模式6`

这组结论和“胜率优先”不同：模式1、模式2更适合研究右尾收益；模式5、模式6虽然部分短周期胜率不低，但在放大止盈目标后收益空间有限。

不同周期胜率汇总：

| 模式 | 策略名 | 5日胜率 | 10日胜率 | 20日胜率 | 40日胜率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `volume_top_pre_breakout` | 46.49% | 46.08% | 47.77% | **49.07%** |
| 2 | `volume_top_breakout` | 42.57% | 43.43% | **43.69%** | 42.70% |
| 3 | `volume_top_follow_through` | 45.98% | 44.71% | **46.02%** | 45.22% |
| 4 | `platform_breakout`（旧模式4） | 38.24% | 37.62% | **41.01%** | 37.08% |
| 5 | `trend_pullback` | **48.14%** | 48.07% | 46.35% | 45.01% |
| 6 | `double_volume_support_rebound` | **47.80%** | 47.38% | 42.95% | 42.45% |

综合排名：

| 排名 | 模式 | 策略名 | 最佳周期胜率 | 综合判断 | 建议定位 |
| ---: | --- | --- | ---: | --- | --- |
| 1 | 模式5 | `trend_pullback` | **48.14%（5日）** | 样本数最大，5/10 日胜率都接近 48%，短周期稳定性最好 | 首选短线观察模式，重点看 5-10 日 |
| 2 | 模式1 | `volume_top_pre_breakout` | **49.07%（40日）** | 纯模式里 40 日胜率最高，平均收益随持有期拉长改善 | 更适合 20-40 日前高预突破潜伏 |
| 3 | 模式6 | `double_volume_support_rebound` | **47.80%（5日）** | 5/10 日表现接近模式5，但 20/40 日明显走弱 | 适合短周期反弹观察，不宜拉长持有 |
| 4 | 模式3 | `volume_top_follow_through` | **46.02%（20日）** | 收益右尾存在，但胜率中等，稳定性不如模式1/5/6 | 作为突破后延续或回踩的补充观察 |
| 5 | 模式2 | `volume_top_breakout` | **43.69%（20日）** | 刚突破后短线胜率偏低，5/10 日平均收益为负 | 不宜单独作为买点，需叠加过滤条件 |
| 6 | 模式4 | `platform_breakout`（旧模式4） | **41.01%（20日）** | 旧平台突破口径在当时六类中最弱；新 `duck_nostril_cross` 需重新回测 | 暂按试错低吸观察，等待新回测 |

| 模式 | 策略名 | 持有天数 | 样本数 | 胜率 | 平均收益 | 平均最大浮盈 | 平均最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `volume_top_pre_breakout` | 5 | 3650 | 46.49% | 0.44% | 6.25% | 4.56% |
| 1 | `volume_top_pre_breakout` | 10 | 3618 | 46.08% | 0.76% | 9.09% | 6.15% |
| 1 | `volume_top_pre_breakout` | 20 | 3582 | 47.77% | 1.40% | 12.95% | 8.15% |
| 1 | `volume_top_pre_breakout` | 40 | 3436 | 49.07% | 2.83% | 18.44% | 10.36% |
| 2 | `volume_top_breakout` | 5 | 1548 | 42.57% | -0.11% | 8.34% | 6.32% |
| 2 | `volume_top_breakout` | 10 | 1529 | 43.43% | -0.11% | 11.41% | 8.11% |
| 2 | `volume_top_breakout` | 20 | 1513 | 43.69% | 0.27% | 15.35% | 10.24% |
| 2 | `volume_top_breakout` | 40 | 1438 | 42.70% | 1.58% | 21.60% | 12.86% |
| 3 | `volume_top_follow_through` | 5 | 3521 | 45.98% | 0.37% | 6.96% | 4.98% |
| 3 | `volume_top_follow_through` | 10 | 3489 | 44.71% | 0.48% | 9.84% | 6.69% |
| 3 | `volume_top_follow_through` | 20 | 3446 | 46.02% | 1.34% | 13.92% | 8.83% |
| 3 | `volume_top_follow_through` | 40 | 3273 | 45.22% | 2.87% | 20.49% | 11.49% |
| 4 | `platform_breakout`（旧模式4） | 5 | 2785 | 38.24% | -0.56% | 5.92% | 5.09% |
| 4 | `platform_breakout`（旧模式4） | 10 | 2783 | 37.62% | -0.89% | 8.31% | 7.15% |
| 4 | `platform_breakout`（旧模式4） | 20 | 2780 | 41.01% | 0.30% | 12.38% | 9.54% |
| 4 | `platform_breakout`（旧模式4） | 40 | 2686 | 37.08% | 0.46% | 18.42% | 12.14% |
| 5 | `trend_pullback` | 5 | 10387 | 48.14% | 0.44% | 5.81% | 4.44% |
| 5 | `trend_pullback` | 10 | 10357 | 48.07% | 0.83% | 8.77% | 6.34% |
| 5 | `trend_pullback` | 20 | 10276 | 46.35% | 1.41% | 13.23% | 8.63% |
| 5 | `trend_pullback` | 40 | 9729 | 45.01% | 2.52% | 19.16% | 11.73% |
| 6 | `double_volume_support_rebound` | 5 | 2866 | 47.80% | 0.37% | 6.71% | 5.06% |
| 6 | `double_volume_support_rebound` | 10 | 2845 | 47.38% | 0.55% | 9.78% | 7.07% |
| 6 | `double_volume_support_rebound` | 20 | 2803 | 42.95% | 0.09% | 14.00% | 10.14% |
| 6 | `double_volume_support_rebound` | 40 | 2709 | 42.45% | 1.78% | 20.51% | 13.57% |

按平均最大浮盈排名，数值越高代表持有期内曾经给出的上冲空间越大：

| 持有周期 | 排名 |
| --- | --- |
| 5日 | **模式2（8.34%）** > 模式3（6.96%） > 模式6（6.71%） > 模式1（6.25%） > 模式4（5.92%） > 模式5（5.81%） |
| 10日 | **模式2（11.41%）** > 模式3（9.84%） > 模式6（9.78%） > 模式1（9.09%） > 模式5（8.77%） > 模式4（8.31%） |
| 20日 | **模式2（15.35%）** > 模式6（14.00%） > 模式3（13.92%） > 模式5（13.23%） > 模式1（12.95%） > 模式4（12.38%） |
| 40日 | **模式2（21.60%）** > 模式6（20.51%） > 模式3（20.49%） > 模式5（19.16%） > 模式1（18.44%） > 模式4（18.42%） |

按平均最大回撤排名，数值越低代表持有期内承受的平均下行压力越小：

| 持有周期 | 排名 |
| --- | --- |
| 5日 | **模式5（4.44%）** > 模式1（4.56%） > 模式3（4.98%） > 模式6（5.06%） > 模式4（5.09%） > 模式2（6.32%） |
| 10日 | **模式1（6.15%）** > 模式5（6.34%） > 模式3（6.69%） > 模式6（7.07%） > 模式4（7.15%） > 模式2（8.11%） |
| 20日 | **模式1（8.15%）** > 模式5（8.63%） > 模式3（8.83%） > 模式4（9.54%） > 模式6（10.14%） > 模式2（10.24%） |
| 40日 | **模式1（10.36%）** > 模式3（11.49%） > 模式5（11.73%） > 模式4（12.14%） > 模式2（12.86%） > 模式6（13.57%） |

模式6在本次补充回测中已生成完整持有样本，短周期表现优于 20/40 日，内部最佳结果为 5 日胜率 `47.80%`。

### `plot`

作用：

- 绘制单只股票 K 线和成交量图

常用示例：

```bash
python -m stocks_analyzer --project-root . plot 603588
python -m stocks_analyzer --project-root . plot 603588 --start-date 20240101 --end-date 20260410
python -m stocks_analyzer --project-root . plot 603588 --output reports/plots/603588_custom.png
```

### `report`

作用：

- 读取已保存的 pattern 结果

常用示例：

```bash
python -m stocks_analyzer --project-root . report --date 2026-04-10
python -m stocks_analyzer --project-root . report --date 2026-04-10 --limit 30
```

### `tradingview`

作用：

- 计算指定日期最近 5 个交易日的 TradingView 风格技术评分

常用示例：

```bash
python -m stocks_analyzer --project-root . tradingview --date 2026-04-10
python -m stocks_analyzer --project-root . tradingview --date 2026-04-10 --top-n 30
python -m stocks_analyzer --project-root . tradingview --date 2026-04-10 --output reports/tradingview/custom.csv
```

默认输出：

- `reports/tradingview/tradingview_avg5_YYYY-MM-DD.csv`
- 最近 5 个交易日的逐日评分文件

### `macd`

作用：

- 生成指定日期的统一技术状态表
- 输出 MACD 金叉/死叉、MACD 顶背离/底背离、量价背离

常用示例：

```bash
python -m stocks_analyzer --project-root . macd --date 2026-04-10
python -m stocks_analyzer --project-root . macd --date 2026-04-10 --top-n 30
python -m stocks_analyzer --project-root . macd --date 2026-04-10 --output reports/macd/custom.csv
```

默认输出：

- `reports/macd/macd_YYYY-MM-DD.csv`
- `reports/macd/macd_YYYY-MM-DD.json`

### `atr`

作用：

- 生成指定日期的 ATR 风险辅助表
- 输出 `ATR14`、`ATR%`、止损止盈参考价和波动分层

常用示例：

```bash
python -m stocks_analyzer --project-root . atr --date 2026-04-10
python -m stocks_analyzer --project-root . atr --date 2026-04-10 --top-n 30
python -m stocks_analyzer --project-root . atr --date 2026-04-10 --output reports/atr/custom.csv
```

默认输出：

- `reports/atr/atr_YYYY-MM-DD.csv`
- `reports/atr/atr_YYYY-MM-DD.json`

### `daily-screening`

作用：

- 判断指定日期是否为交易日
- 串行执行 `update -> tradingview -> macd -> atr -> trend-universe -> trend -> pattern`
- `trend` 生成 `watchlist_trend`
- `pattern` 生成 `watchlist_pattern`
- 兼容保留一份通用 `watchlist`，当前与 `watchlist_pattern` 同步
- 写入运行摘要

常用示例：

```bash
python -m stocks_analyzer --project-root . daily-screening --date 2026-04-10
python -m stocks_analyzer --project-root . daily-screening --date 2026-04-10 --start-date 20240101
```

主要参数：

- `--date`：目标日期，格式 `YYYY-MM-DD`
- `--start-date`：更新数据起始日期，格式 `YYYYMMDD`

当前执行顺序：

1. 判断交易日
2. `update`
3. `tradingview`
4. `macd`
5. `atr`
6. `trend-universe`
7. `trend`
8. `pattern`
9. 生成 `watchlist_pattern`、`watchlist_trend` 和兼容用 `watchlist`
10. 写入 `daily_screening_YYYY-MM-DD.json`

默认输出：

- `reports/watchlists/watchlist_YYYY-MM-DD.json`
- `reports/watchlists/watchlist_pattern_YYYY-MM-DD.json`
- `reports/watchlists/watchlist_trend_YYYY-MM-DD.json`
- `reports/daily_screening/daily_screening_YYYY-MM-DD.json`
- `reports/patterns/patterns_all_YYYY-MM-DD.csv`
- `reports/tradingview/tradingview_avg5_YYYY-MM-DD.csv`
- `reports/macd/macd_YYYY-MM-DD.csv`
- `reports/atr/atr_YYYY-MM-DD.csv`
- `reports/trend/trend_YYYY-MM-DD.csv`
- `reports/trend_universe/trend_universe_YYYY-MM-DD.csv`

重要说明：

- `daily-screening` **不会修改** `选股.md`
- 同一日期重复执行时，会覆盖同日期的 `watchlist` 和运行摘要
- `watchlist_pattern` 不再强制要求 `trend-universe` 交集
- `watchlist_pattern` 和 `watchlist_trend` 都会先剔除 `MACD顶背离 / 量价看空 / dead_cross`
- `watchlist_trend` 只保留 `buy_score / price_action_score` 高于 `pick_trend_watchlist` 阈值的票
- 主 `watchlist` 和 `watchlist_pattern` 会补入 `ATR14`、`ATR%`、止损止盈参考、`波动分层`
- 主 `watchlist` 和 `watchlist_pattern` 会补入 `连续上榜天数`，按主 `watchlist_YYYY-MM-DD.json` 连续出现天数累计
- `patterns_all_YYYY-MM-DD.csv` 会同时补入第一层趋势字段和 `trend` 评分字段，方便后续 `pick`

### `trend`

作用：

- 扫描全市场并输出指定日期的趋势复核结果
- 输出 `buy_score`、`price_action_score`
- 同时保留 `MACD` 金叉死叉、MACD 背离、量价背离等展示字段

常用示例：

```bash
python -m stocks_analyzer --project-root . trend --date 2026-04-10
python -m stocks_analyzer --project-root . trend --date 2026-04-10 --top-n 50
```

默认输出：

- `reports/trend/trend_YYYY-MM-DD.csv`
- `reports/trend/trend_YYYY-MM-DD.json`

### `intraday-screening`

作用：

- 读取上一交易日或指定日期的 `watchlist`
- 运行时抓取这些股票“当日 5 分钟线”
- 基于 `watchlist` 中已有的日线辅助字段和当日 5 分钟线事件生成盘中排序结果

常用示例：

```bash
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11 --watchlist-date 2026-04-10
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11 --start-date 20240101 --top-n 30
```

主要参数：

- `--date`：目标日期，格式 `YYYY-MM-DD`
- `--watchlist-date`：指定使用哪一天的 `watchlist`
- `--start-date`：更新数据起始日期，格式 `YYYYMMDD`
- `--top-n`：终端展示前 N 行

### `trend-universe`

作用：

- 基于本地日线数据定义趋势股池
- 使用“先硬过滤，再趋势评分”的方式输出 `trend universe`

常用示例：

```bash
python -m stocks_analyzer --project-root . trend-universe --date 2026-04-10
python -m stocks_analyzer --project-root . trend-universe --date 2026-04-10 --top-n 50
```

### `trend-signals`

作用：

- 在 `trend universe` 上识别 `breakout` 和 `pullback` 两类入场信号

常用示例：

```bash
python -m stocks_analyzer --project-root . trend-signals --date 2026-04-10
python -m stocks_analyzer --project-root . trend-signals --date 2026-04-10 --top-n 50
```

### `trend-score`

作用：

- 在 `breakout/pullback` setup 上叠加 `均线/ADX + MACD + RSI + BOLL + 成交量 + KDJ + ATR`
- 输出收盘后的多指标评分，包括 `buy_score`
- `MACD` 除了金叉死叉，还会纳入顶背离减分、底背离加分
- `量价背离` 单独计分，不和普通量能项混在一起
- `price_action_score` 是触发分里权重最高的分项

常用示例：

```bash
python -m stocks_analyzer --project-root . trend-score --date 2026-04-10
```

### `trend-entries`

作用：

- 基于收盘评分输出“次日开盘买入”的候选
- 默认会按配置里的门槛过滤，当前默认规则是：
  - `buy_score >= 80`
  - `trend_base_score >= 65`
  - `price_action_score >= 60`
  - `macd_score >= 35`
  - `positive_indicator_count >= 3`
- 默认门槛来自一轮真实数据抽样回测，不是固定真理，只是第一版研究默认值
- `trend-entries` 只输出已经过上述过滤的可交易候选；如果你想看全部评分分布，用 `trend-score`

常用示例：

```bash
python -m stocks_analyzer --project-root . trend-entries --date 2026-04-10
```

### `backtest-signals`

作用：

- 对趋势信号执行固定持有 `5/10/20/40` 日的单信号回测
- 第一版按“当日收盘确认并按当日收盘入场”的研究口径统计

常用示例：

```bash
python -m stocks_analyzer --project-root . backtest-signals --date 2026-04-10 --start-date 2025-01-01
```

### `backtest-portfolio`

作用：

- 每天按评分选择前 `N` 只趋势信号
- 做组合层前 N 等权、固定持有的基线回测

常用示例：

```bash
python -m stocks_analyzer --project-root . backtest-portfolio --date 2026-04-10 --start-date 2025-01-01
```

### `backtest-entries`

作用：

- 对收盘评分后的候选执行 `next_open` 回测
- 即 `t` 日收盘生成评分，`t+1` 日开盘买入
- 回测对象与 `trend-entries` 一样，会先经过默认门槛过滤
- 这是当前更接近真实执行的研究口径，和 `backtest-signals` 的 `same_close` 口径不同

常用示例：

```bash
python -m stocks_analyzer --project-root . backtest-entries --date 2026-04-10 --start-date 2025-01-01
```

### `backtest-entries-portfolio`

作用：

- 基于 `buy_score` 执行次日开盘入场的组合回测
- 每天按 `buy_score` 排序后取前 `N` 只等权持有
- 同样先经过默认门槛过滤

常用示例：

```bash
python -m stocks_analyzer --project-root . backtest-entries-portfolio --date 2026-04-10 --start-date 2025-01-01
```

### `research-thresholds`

作用：

- 在不给 `buy_score` 预设最终交易阈值的前提下，先研究历史样本分布
- 样本只保留基础初筛条件：进入 `trend universe`、识别出 `breakout/pullback` setup、且次日存在可交易开盘价
- 对每条样本按 `next_open` 口径回测 `5/10/20/40` 日表现
- 比较强势组、弱势组和底部组在各个评分项上的分布差异
- 生成单指标候选阈值和组合阈值对比表

主要参数：

- `--date`：研究截止日期，格式 `YYYY-MM-DD`
- `--start-date`：研究开始日期，格式 `YYYY-MM-DD`
- `--sample-mode`：历史截面抽样方式，可选 `daily`、`weekly`、`monthly`
- `--train-end-date`：可选，样本内结束日期；传入后会同时输出 `all_period`、`in_sample`、`out_of_sample`
- `--output`：可选，自定义样本明细 CSV 输出路径

常用示例：

```bash
python -m stocks_analyzer --project-root . research-thresholds --date 2026-04-10 --start-date 2025-01-01 --sample-mode monthly
python -m stocks_analyzer --project-root . research-thresholds --date 2026-04-10 --start-date 2024-01-01 --sample-mode weekly --train-end-date 2025-06-30
```

## 趋势评分说明

当前多指标买入评分按两层结构计算：

- `trend_base_score`：回答“这只票本身是不是适合做日 K 趋势交易”
- `trigger_score`：回答“这一天是不是一个足够强的买点”

总分结构：

```text
buy_score = trend_base_score * 0.35 + trigger_score * 0.65
```

其中 `trigger_score` 由以下分项加权得到：

- `price_action_score`：权重最高，承载 `breakout/pullback` 结构质量
- `macd_score`：包含金叉死叉，也包含顶背离减分、底背离加分
- `volume_score`
- `volume_price_divergence_score`
- `boll_score`
- `rsi_score`
- `kdj_score`
- `atr_score`

当前默认阈值：

- `buy_score >= 80`
- `trend_base_score >= 65`
- `price_action_score >= 60`
- `macd_score >= 35`
- `positive_indicator_count >= 3`

当前代码里还额外启用了分信号覆盖规则：

- `breakout` 默认使用更严格的研究阈值：
  - `buy_score >= 81.3308`
  - `price_action_score >= 75.0373`
- `pullback` 暂时继续沿用上面的全局默认阈值

这组默认值用于第一版研究回测，目的是在信号质量和样本数量之间先取一个中间点。后面更合理的做法，是继续按你的数据区间回测，再迭代这些阈值和权重。

## 趋势链路输出

趋势链路会输出独立于 `pattern/watchlist` 的结果，不会覆盖旧文件。

- `trend-universe`：输出趋势股池明细、汇总和 JSON 摘要
- `trend-signals`：输出 `breakout/pullback` setup 明细、汇总和 JSON 摘要
- `trend-score`：输出收盘评分明细、汇总和 JSON 摘要
- `trend-entries`：输出次日开盘买入候选明细、汇总和 JSON 摘要
- `backtest-signals`：输出 setup 层回测明细和汇总
- `backtest-portfolio`：输出 setup 层组合持仓、净值和汇总
- `backtest-entries`：输出次日开盘单信号回测明细和汇总
- `backtest-entries-portfolio`：输出次日开盘组合持仓、净值和汇总
- `research-thresholds`：输出阈值研究样本、分布对比、候选阈值、单指标阈值回测和组合阈值回测

## `intraday-screening` 输出

默认输出目录：

- `reports/intraday_screening/YYYY-MM-DD/`

典型输出文件：

- `intraday_screening_YYYY-MM-DD.json`
- `intraday_rank_YYYY-MM-DD.csv`

`intraday_rank_YYYY-MM-DD.csv` 的特点：

- 按 `5分钟分数` 降序排序
- 表头为中文，便于直接打开查看
- 同时列出 `形态`、TradingView 分数、日线背离、`5分钟分数`
- 会把量价背离、5 分钟 MACD 背离、金叉死叉、均线事件的命中情况和类型一起写出

重要说明：

- `intraday-screening` **不会修改** `选股.md`
- `intraday-screening` **不会生成新的 `watchlist`**
- `intraday-screening` 当前**不会更新日线 parquet**
- `intraday-screening` 当前**不会把 5 分钟 K 线长期缓存到本地 parquet**；5 分钟数据是在运行时抓取并用于计算盘中分数
- 当前建议的日线源顺序是：先 `sina`，如果 `sina` 不稳定，再尝试 `baostock`
- 分钟线数据源 `intraday_provider` 目前建议使用 `itick`
- 仓库里保留了 `tushare` 分钟线 provider 代码，后续如果你要切换，只需要把 `intraday_provider` 改成 `tushare` 并配置环境变量 `TUSHARE_TOKEN`
- 现在也支持把 `intraday_provider` 改成 `itick`；使用前需要配置环境变量 `ITICK_TOKEN`
- 在 `itick` 模式下，盘中复筛会优先走批量 `/stock/klines` 请求，尽量降低 `5 次/分钟` 配额下的限流风险
- 如果个别股票的 5 分钟线抓取失败，盘中复筛会在 JSON 摘要里记录失败股票并跳过，不会中断整轮
- 同一日期重复执行时，会覆盖同日期的盘中结果文件

### `train-prob`

作用：

- 基于本地主板日线数据构建样本
- 训练中短期上涨概率模型

常用示例：

```bash
python -m stocks_analyzer --project-root . train-prob
python -m stocks_analyzer --project-root . train-prob --start-date 2023-01-01 --end-date 2025-12-31
python -m stocks_analyzer --project-root . train-prob --limit 500
```

### `predict-prob`

作用：

- 读取已训练模型，对指定日期生成全市场概率排序

常用示例：

```bash
python -m stocks_analyzer --project-root . predict-prob --date 2026-04-10
python -m stocks_analyzer --project-root . predict-prob --date 2026-04-10 --top-n 30
```

### `xueqiu-archive`

作用：

- 归档雪球博主 `1155695148` 的公开历史帖子

常用示例：

```bash
python -m stocks_analyzer --project-root . xueqiu-archive --headed --refresh
python -m stocks_analyzer --project-root . xueqiu-archive --max-posts 20
python -m stocks_analyzer --project-root . xueqiu-archive --output reports/xueqiu/custom.md
```

## watchlist 的生成规则

当前有两套候选池：

- `watchlist_pattern`：来自 `pattern`，按原有技术规则筛出候选
- `watchlist_trend`：来自 `trend`，只保留高分趋势 setup

两套候选池都会先做一层风险剔除：

- `MACD顶背离`
- `量价看空`
- `dead_cross`

其中 `watchlist_trend` 还会额外要求：

- `buy_score` 达到 `pick_trend_watchlist.buy_score_min`
- `price_action_score` 达到 `pick_trend_watchlist.price_action_score_min`

核心规则包括：

- 只保留 `TradingView` 标签为 `buy` 或 `strong_buy` 的标的
- 不同 `pattern_id` 使用不同的 5 日均分阈值
- 过滤指数类名称
- 计算 `stable_score`
- 输出 `第一梯队 / 第二梯队 / 第三梯队`
- 最终按梯队、`stable_score`、`tradingview_avg_5d` 排序

`watchlist_pattern` / `watchlist_trend` 适合做：

- 次日盘中复筛的输入
- 你自己手工选股时的技术候选池

`watchlist` 不等于：

- 最终买入名单
- `选股.md`

## 推荐的完整选股流程

推荐把项目当作“技术筛选引擎”，而不是自动选股器。

### 1. 初始化本地数据

```bash
python -m stocks_analyzer --project-root . update --start-date 20240101
```

### 2. 日终生成技术候选池

```bash
python -m stocks_analyzer --project-root . daily-screening --date 2026-04-10
```

这一步会得到：

- 当日全市场技术结果
- 当日 `watchlist`

### 3. 次日盘中做小范围复筛

```bash
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11
```

如果需要指定来源：

```bash
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11 --watchlist-date 2026-04-10
```

### 4. 你自己做最终决策

推荐结合以下材料：

- 前一日 `watchlist`
- 当日 `intraday-screening` 结果
- `reports/intraday_screening/<date>/intraday_rank_<date>.csv`
- 你自己的 `主线.md`
- 你自己的 `选股.md`

## 配置

默认配置文件：

- [`config/default.yaml`](/C:/Users/wdyab/Desktop/wdy/stocks/config/default.yaml)

主要可调项：

- 数据源 `provider`
- 分钟线数据源 `intraday_provider`
- 复权方式 `adjustment`
- 流动性门槛
- pattern 阈值
- 网络代理配置

## 注意事项

- `daily-screening` 更适合盘后或定时任务，不适合盘中全市场快速执行
- `intraday-screening` 依赖已有 `watchlist`
- 如果没有可用 `watchlist`，盘中复筛会报错
- 当前推荐的数据源选择是：先 `sina`，如果 `sina` 不稳定，再尝试 `baostock`
- 交易日判断当前优先走 `akshare` 的交易日历接口，失败后退化到工作日判断
- 雪球归档属于最佳努力抓取，不保证完整覆盖

## 兼容入口

当前主入口建议统一使用：

```bash
python -m stocks_analyzer --project-root . <subcommand>
```

也兼容：

- `mystock`
- `stocks-analyzer`

## `runcmd` 固定入口

如果你希望通过对话只修改一个本地文本文件，然后在终端里始终执行同一个固定命令，可以使用项目内置的 `runcmd` 方案。

约定如下：

- 命令文件固定为项目根目录的 `command.txt`
- 执行器脚本固定为 [`tools/runcmd.py`](/C:/Users/wdyab/Desktop/wdy/stocks/tools/runcmd.py)
- `command.txt` 建议只放一条完整 PowerShell 命令

例如，`command.txt` 可以写成：

```powershell
python -m stocks_analyzer --project-root . daily-screening --date 2026-04-14
```

直接运行脚本的方式：

```powershell
python .\tools\runcmd.py
```

如果你希望以后只输入 `runcmd`，更实用的做法是在 PowerShell 配置文件里加一个函数：

```powershell
function runcmd { python "C:\Users\wdyab\Desktop\wdy\stocks\tools\runcmd.py" }
```

写入配置文件的一种方式：

```powershell
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
Add-Content $PROFILE 'function runcmd { python "C:\Users\wdyab\Desktop\wdy\stocks\tools\runcmd.py" }'
```

重开一个 PowerShell 窗口后，你就可以固定使用：

```powershell
runcmd
```

运行行为：

- `runcmd` 会读取项目根目录的 `command.txt`
- 如果文件不存在或为空，会直接报错退出
- 执行前会先打印当前将要执行的命令
- 实际命令始终在项目根目录下运行
- 执行结束后不会清空 `command.txt`
