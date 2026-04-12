# A Share Analyzer

面向 A 股主板的命令行技术面分析框架。V1 聚焦分析，不包含交易执行。

## 当前能力

- 主板股票池更新
- 日线行情抓取与本地缓存
- 4 类日 K 趋势模式识别
- TradingView 风格 26 指标 Technical Ratings
- 交易日感知的每日自动筛选与 Markdown 归档
- 基于日线的中短期上涨概率训练与全市场排序
- 雪球博主 1155695148 公开历史帖子 Markdown 归档
- 单股 K 线加成交量作图
- CSV 结果导出和命令行展示

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install playwright
mystock --help
```

## 常用命令

```bash
mystock update --start-date 20240101
mystock update --start-date 20240101 --skip-existing
mystock update 603588 --start-date 20240101
mystock pattern
mystock pattern --1
mystock pattern --2 --4
mystock pattern --plot-all
mystock tradingview --date 2026-04-10
mystock daily-screening --date 2026-04-10
mystock train-prob
mystock predict-prob --date 2026-04-10
mystock xueqiu-archive --headed --refresh
mystock plot 603588
mystock report --date 2026-04-10
```

兼容入口 `python main.py ...` 和 `python plot_symbol.py ...` 仍然可用，但主入口已经统一为 `mystock`。

`mystock update` 在不传股票代码时会先刷新主板股票池，再批量更新日线数据；传入单只股票代码时，只更新该股票。
`mystock pattern` 默认识别全部 4 个模式，也可以用 `--1 --2 --3 --4` 按需指定。
`mystock pattern` 会生成 `reports/patterns/patterns_all_YYYY-MM-DD.csv` 或 `reports/patterns/patterns_1_YYYY-MM-DD.csv` 这类 CSV，只包含命中模式的股票。
`mystock pattern --plot-all` 会额外为所有命中股票生成图形，输出到 `reports/plots/YYYY-MM-DD/`。
`mystock tradingview` 会计算最近 5 个交易日的 TradingView 总分，按 5 日平均总分排序，并额外保存这 5 个交易日各自的评分 CSV。
`mystock daily-screening` 会先判断指定日期是否为交易日；如果是，就串行执行 `update`、`tradingview`、`pattern`，再把最新筛选结果插入 [`选股.md`](C:\Users\wdyab\Desktop\wdy\stocks\选股.md) 顶部。
`mystock train-prob` 会基于本地主板日线构建样本，并训练 XGBoost 模型。
`mystock predict-prob` 会读取已训练模型，对指定日期的主板股票生成概率排序 CSV，输出到 `reports/probability/`。
`mystock xueqiu-archive` 会尝试抓取雪球博主 `1155695148` 的公开历史帖子，并导出到 `reports/xueqiu/1155695148.md`。如果雪球触发滑动验证，建议使用 `--headed` 打开可见浏览器手动完成验证。

## 每日筛选

`mystock daily-screening` 是给日常盘前或定时任务准备的统一入口。
它会把原本分散的几个步骤固定下来，避免每次手工重复执行。

默认执行顺序：

1. 判断指定日期是否为 A 股交易日
2. 如果不是交易日，直接跳过，不更新数据
3. 如果是交易日，执行 `mystock update --start-date 20240101`
4. 执行 `mystock tradingview --date YYYY-MM-DD`
5. 执行 `mystock pattern --as-of YYYY-MM-DD`
6. 调用 `skills/project-stock-picker/scripts/project_stock_picker.py`
7. 生成分梯队结果，并把最新一期插入 [`选股.md`](C:\Users\wdyab\Desktop\wdy\stocks\选股.md) 顶部

常用示例：

```bash
mystock daily-screening --date 2026-04-10
mystock daily-screening --date 2026-04-10 --start-date 20240101
mystock daily-screening --date 2026-04-10 --picks-file 选股.md
```

主要参数：

- `--date`
  - 目标日期，格式 `YYYY-MM-DD`
  - 不传时默认使用当天日期
- `--start-date`
  - 更新日线数据时使用的起始日期，格式 `YYYYMMDD`
  - 默认值是 `20240101`
- `--picks-file`
  - 选股结果 Markdown 文件名
  - 默认写入项目根目录下的 `选股.md`

默认输出：

- 选股 Markdown：[`选股.md`](C:\Users\wdyab\Desktop\wdy\stocks\选股.md)
- 每次运行的简要报告：`reports/daily_screening/daily_screening_YYYY-MM-DD.json`
- 模式文件：`reports/patterns/patterns_all_YYYY-MM-DD.csv`
- TradingView 文件：`reports/tradingview/tradingview_avg5_YYYY-MM-DD.csv`

当前实现说明：

- 当天新增结果会插入在 `选股.md` 最前面，旧记录自动后移。
- 总结段落会结合 `选股.md` 里最近几日的历史结果，做简单趋势描述和重复入选统计。
- `行业/主线` 当前优先复用历史 `选股.md` 中已经出现过的映射；如果是首次出现的新股票，可能暂时记为 `未分类`。
- 交易日判断会优先使用配置里的数据源；如果数据源不可用，会退化到工作日判断。

## 雪球归档

当前仅支持固定账号 `1155695148`。

第一次使用建议：

```bash
pip install playwright
mystock xueqiu-archive --headed --refresh
```

常用参数：

- `--headed`
  - 打开可见浏览器，便于手动完成雪球滑动验证
- `--refresh`
  - 忽略本地链接缓存并重新发现帖子
- `--max-posts 20`
  - 仅抓取前 20 条帖子，便于小范围测试
- `--output reports/xueqiu/custom.md`
  - 自定义 Markdown 输出路径

默认输出和缓存位置：

- 归档结果：`reports/xueqiu/1155695148.md`
- 已发现链接缓存：`data/xueqiu/1155695148/discovered_urls.json`
- 单帖中间结果：`data/xueqiu/1155695148/raw/`
- 浏览器持久化资料目录：`data/xueqiu/1155695148/browser_profile/`

注意事项：

- 这是基于公开页面的最佳努力抓取，不保证覆盖全部历史帖子。
- 如果无头模式下被雪球拦截，命令会提示改用 `--headed`。
- 当前项目中的 Python 请求仍然会按 `config/default.yaml` 里的代理配置运行，但雪球浏览器归档默认不会继承这些代理环境变量，以减少国内站点风控干扰。

## 配置

默认配置文件位于 `config/default.yaml`。4 个模式的阈值、日线复权方式、流动性门槛都可以在这里调整。
当前默认数据源是 `baostock`；如果后面想切回 `akshare`，只需要修改 `provider` 字段。

## 四个模式分别识别什么

### 模式 1

模式 1 用来识别“接近前高型”的股票。
这类股票通常在过去一段时间里出现过一个比较重要的高点，后面经历了一轮明显回撤，现在又重新走强并回到老前高附近。
它更适合拿来观察“是否会重新挑战关键压力位”。

### 模式 2

模式 2 用来识别“平台突破型”的股票。
这类股票往往已经有一定上涨基础，随后在较窄区间内横盘整理，当前价格开始接近或突破平台上沿，并伴随量能改善。
它更像是在找趋势中的再次加速点。

### 模式 3

模式 3 用来识别“趋势回踩型”的股票。
这类股票中期趋势通常已经建立，价格在上涨后回踩到关键均线附近，但整体趋势并没有走坏，回踩过程中的量能也相对温和。
它更适合观察“强趋势中的低风险跟随机会”。

### 模式 4

模式 4 用来识别“强势股二波型”的股票。
这类股票前面通常已经有一段比较明显的快速上涨，随后进入短期整理或降温阶段，现在又出现再次启动的迹象。
它更适合拿来观察“强势股是否会走出第二波”。

## 模式与 TradingView 的关系

基于本地快照 `reports/patterns/patterns_all_2026-04-11.csv` 和 `reports/tradingview/tradingview_avg5_2026-04-11.csv`，可以总结出一个比较稳定的经验规律：

- 模式识别负责回答“形态结构是否成立”。
- TradingView 评分负责回答“当前趋势强度和指标共振是否足够强”。
- 两者叠加后，比单看任一侧更适合做筛选优先级排序。

在 `2026-04-11` 这批数据中：

- 全市场 5 日平均 `avg_all_rating_5d` 均值为 `-0.1377`，中位数为 `-0.2358`
- 模式命中股票的 5 日平均 `tradingview_avg_all_rating_5d` 均值为 `0.3500`，中位数为 `0.3467`
- 全市场 TradingView 标签分布以 `sell` 为主，`sell + strong_sell` 合计约 `50.63%`
- 模式命中股票中，`buy + strong_buy` 占比为 `100%`

这说明当前 4 个模式本质上都不是在全市场里随机捞股票，而是在优先识别“已经具备中短期技术共振”的标的。

进一步按模式拆开看：

- 模式 1
  - 平均 `tradingview_avg_all_rating_5d` 约为 `0.3623`
  - `strong_buy` 占比约 `66.67%`
  - 特点是老前高附近重新转强，往往同时伴随更强的趋势确认，所以在 4 个模式里属于“结构和强度最统一”的一类
- 模式 2
  - 平均 `tradingview_avg_all_rating_5d` 约为 `0.3585`
  - `buy` 占比约 `75%`，`strong_buy` 占比约 `25%`
  - 特点是平台突破很多已经进入共振区，但未必都走到最强趋势加速，因此更常见的是稳定 `buy`，而不是极端强势
- 模式 3
  - 平均 `tradingview_avg_all_rating_5d` 约为 `0.2707`
  - `buy` 占比 `100%`，没有 `strong_buy`
  - 特点是趋势回踩型本来就更偏“顺趋势低吸/跟随”，结构成立时通常仍然偏强，但动量没有模式 1 和模式 4 那么极致
- 模式 4
  - 平均 `tradingview_avg_all_rating_5d` 约为 `0.3893`
  - `strong_buy` 占比约 `40%`
  - 特点是强势股二波本身就要求前期强势和再次启动，因此和 TradingView 强势评分的耦合度也很高

实践上可以这样理解：

- 如果某只股票命中模式，同时 TradingView 为 `strong_buy`，说明“结构”与“强度”同时成立，优先级可以更高
- 如果命中模式但只到 `buy`，通常仍然值得观察，只是更偏早期确认或震荡上行
- 模式 3 即使只有 `buy` 也不一定弱，因为它本身不是追求最强加速，而是追求强趋势中的回踩修复
- 如果以后某次模式命中结果里出现大量 `neutral` 或 `sell`，通常意味着结构虽然勉强成立，但短期共振不足，信号质量应下调

因此，当前项目里更合适的使用方式不是“模式”和 “TradingView” 二选一，而是：

- 先用模式筛掉结构不清晰的股票
- 再用 TradingView 判断这些候选股目前的趋势一致性和强弱排序
- 在同一模式内部，优先关注 `tradingview_avg_all_rating_5d` 更高、且标签达到 `strong_buy` 的标的

## 代理

当前默认已在 [default.yaml](C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml) 中配置：

```yaml
network:
  http_proxy: http://127.0.0.1:7897
  https_proxy: http://127.0.0.1:7897
  no_proxy: 127.0.0.1,localhost
```

如果你后面更换代理端口，只需要修改这三个值，不需要改 Python 代码。
