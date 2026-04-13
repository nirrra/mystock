# A Share Analyzer

面向 A 股主板的命令行技术分析工具。

当前项目聚焦三件事：

- 更新主板股票池和本地日线缓存
- 生成 pattern、TradingView、MACD 背离等技术结果
- 基于技术规则生成 `watchlist`，并在次日盘中做小范围复筛

项目不包含交易执行，也不会替你自动决定最终买入标的。

## 当前定位

当前流程里有两层结果：

- 技术结果层：`pattern`、`tradingview`、`divergence`
- 候选池层：`watchlist`

其中：

- `daily-screening` 负责跑完整技术链路，并生成当日 `watchlist`
- `intraday-screening` 负责读取上一交易日或指定日期的 `watchlist`，只对候选股做盘中复筛
- `选股.md` 不在自动命令链里更新，留给你手动整理和最终决策

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
reports/divergence/          MACD 背离结果
reports/watchlists/          日终 watchlist
reports/daily_screening/     daily-screening 运行摘要
reports/intraday_screening/  盘中复筛结果
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
python -m stocks_analyzer --project-root . divergence --date 2026-04-10
```

执行一轮完整日终筛选并生成 `watchlist`：

```bash
python -m stocks_analyzer --project-root . daily-screening --date 2026-04-10
```

次日盘中只对 `watchlist` 做复筛：

```bash
python -m stocks_analyzer --project-root . intraday-screening --date 2026-04-11
```

## 命令说明

### `update`

作用：

- 不传股票代码时，刷新主板股票池并批量更新日线数据
- 传入股票代码时，只更新单只股票

常用示例：

```bash
python -m stocks_analyzer --project-root . update --start-date 20240101
python -m stocks_analyzer --project-root . update --start-date 20240101 --skip-existing
python -m stocks_analyzer --project-root . update 603588 --start-date 20240101
python -m stocks_analyzer --project-root . update --start-date 20240101 --end-date 20260413 --limit 100
```

主要参数：

- `symbol`：可选，6 位股票代码
- `--start-date`：开始日期，格式 `YYYYMMDD`
- `--end-date`：结束日期，格式 `YYYYMMDD`
- `--limit`：只更新前 N 只股票
- `--skip-existing`：跳过本地已有缓存

### `pattern`

作用：

- 扫描本地日线缓存，识别 1 到 4 号模式

常用示例：

```bash
python -m stocks_analyzer --project-root . pattern
python -m stocks_analyzer --project-root . pattern --1
python -m stocks_analyzer --project-root . pattern --2 --4
python -m stocks_analyzer --project-root . pattern --as-of 2026-04-10 --output reports/my_patterns.csv
python -m stocks_analyzer --project-root . pattern --plot-all
```

主要参数：

- `--1 --2 --3 --4`：只识别指定模式
- `--as-of`：分析截止日期，格式 `YYYY-MM-DD`
- `--limit`：终端展示上限
- `--output`：自定义 CSV 输出路径
- `--plot-all`：为命中股票批量生成图形

默认输出：

- `reports/patterns/patterns_all_YYYY-MM-DD.csv`

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

### `divergence`

作用：

- 识别指定日期最近 15 个交易日内的 MACD 顶背离和底背离

常用示例：

```bash
python -m stocks_analyzer --project-root . divergence --date 2026-04-10
python -m stocks_analyzer --project-root . divergence --date 2026-04-10 --top-n 30
python -m stocks_analyzer --project-root . divergence --date 2026-04-10 --output reports/divergence/custom.csv
```

默认输出：

- `reports/divergence/macd_divergence_YYYY-MM-DD.csv`

### `daily-screening`

作用：

- 判断指定日期是否为交易日
- 串行执行 `update -> tradingview -> divergence -> pattern`
- 基于技术规则生成当日 `watchlist`
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
4. `divergence`
5. `pattern`
6. 生成 `watchlist`
7. 写入 `daily_screening_YYYY-MM-DD.json`

默认输出：

- `reports/watchlists/watchlist_YYYY-MM-DD.json`
- `reports/daily_screening/daily_screening_YYYY-MM-DD.json`
- `reports/patterns/patterns_all_YYYY-MM-DD.csv`
- `reports/tradingview/tradingview_avg5_YYYY-MM-DD.csv`
- `reports/divergence/macd_divergence_YYYY-MM-DD.csv`

重要说明：

- `daily-screening` **不会修改** `选股.md`
- 同一日期重复执行时，会覆盖同日期的 `watchlist` 和运行摘要

### `intraday-screening`

作用：

- 读取上一交易日或指定日期的 `watchlist`
- 只更新 `watchlist` 中的股票
- 只对这些股票执行 `tradingview / divergence / pattern`

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

默认输出目录：

- `reports/intraday_screening/YYYY-MM-DD/`

典型输出文件：

- `intraday_screening_YYYY-MM-DD.json`
- `tradingview_avg5_YYYY-MM-DD.csv`
- `macd_divergence_YYYY-MM-DD.csv`
- `patterns_all_YYYY-MM-DD.csv`

重要说明：

- `intraday-screening` **不会修改** `选股.md`
- `intraday-screening` **不会生成新的 `watchlist`**
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

当前 `watchlist` 仍然沿用原来的技术筛选规则，不依赖人工主线判断，也不依赖 GPT。

核心规则包括：

- 只保留 `TradingView` 标签为 `buy` 或 `strong_buy` 的标的
- 不同 `pattern_id` 使用不同的 5 日均分阈值
- 过滤指数类名称
- 计算 `stable_score`
- 输出 `第一梯队 / 第二梯队 / 第三梯队`
- 最终按梯队、`stable_score`、`tradingview_avg_5d` 排序

`watchlist` 适合做：

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
- 你自己的 `主线.md`
- 你自己的 `选股.md`

## 配置

默认配置文件：

- [`config/default.yaml`](/C:/Users/wdyab/Desktop/wdy/stocks/config/default.yaml)

主要可调项：

- 数据源 `provider`
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
