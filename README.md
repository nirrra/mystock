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

- 扫描本地日线缓存，识别 1 到 5 号模式

常用示例：

```bash
python -m stocks_analyzer --project-root . pattern
python -m stocks_analyzer --project-root . pattern --1
python -m stocks_analyzer --project-root . pattern --2 --5
python -m stocks_analyzer --project-root . pattern --as-of 2026-04-10 --output reports/my_patterns.csv
python -m stocks_analyzer --project-root . pattern --plot-all
```

主要参数：

- `--1 --2 --3 --4 --5`：只识别指定模式
- `--as-of`：分析截止日期，格式 `YYYY-MM-DD`
- `--limit`：终端展示上限
- `--output`：自定义 CSV 输出路径
- `--plot-all`：为命中股票批量生成图形

默认输出：

- `reports/patterns/patterns_all_YYYY-MM-DD.csv`

#### 五个模式分别识别什么

##### 模式1：量顶天立地预突破型

这类股票通常在过去一段时间里出现过一个明显前高，之后经历了较长时间的回撤修复，当前仍未突破，但已经重新逼近老前高。

它更适合拿来观察“第二天是否可能放量突破前高”。

##### 模式2：量顶天立地突破确认型

这类股票在完成底部修复后，最新一根日线已经出现放量阳线，并且最高价突破了前高。

它更适合拿来观察“突破后的次日跟踪机会”。

##### 模式3：量顶天立地突破后延续/回踩型

这类股票已经在前几天完成了量顶天立地式突破，目前仍处在突破后 1 到 8 个交易日内，既可能直接延续，也可能回踩 `MA20` 后重新站回。

它更适合拿来观察“突破后的二次上车机会”。

##### 模式4：平台突破型

这类股票前面先走出过一段比较明确的主升，随后经过 1 到 3 天过渡，进入至少 20 个交易日的平台整理。最近几天里，它开始放量突破平台上沿，而且当前离平台高点还不算太远。

它更像是在找“第一段主升之后，平台蓄势，再次准备打开空间”的票。

##### 模式5：趋势回踩型

这类股票近期刚打出过一个短期高点，随后在最近两天内回踩到 `MA20` 附近甚至盘中短破，但收盘又重新站回 `MA20` 上方，同时 `MA20` 仍然保持在 `MA60` 之上。

它更适合观察“强趋势里的深洗盘后，是否会重新转强”。

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
- 默认配置下，日线数据源 `provider` 使用 `baostock`，分钟线数据源 `intraday_provider` 使用 `itick`
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
- 交易日判断会优先尝试数据源接口，失败后退化到工作日判断
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
