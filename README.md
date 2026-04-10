# A Share Analyzer

面向 A 股主板的命令行技术面分析框架。V1 聚焦分析，不包含交易执行。

## 当前能力

- 主板股票池更新
- 日线行情抓取与本地缓存
- 4 类日 K 趋势模式识别
- 单股 K 线加成交量作图
- CSV 结果导出和命令行展示

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
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
mystock plot 603588
mystock report --date 2026-04-10
```

兼容入口 `python main.py ...` 和 `python plot_symbol.py ...` 仍然可用，但主入口已经统一为 `mystock`。

`mystock update` 在不传股票代码时会先刷新主板股票池，再批量更新日线数据；传入单只股票代码时，只更新该股票。
`mystock pattern` 默认识别全部 4 个模式，也可以用 `--1 --2 --3 --4` 按需指定。
`mystock pattern` 会生成 `reports/patterns_all_YYYY-MM-DD.csv` 或 `patterns_1_YYYY-MM-DD.csv` 这类 CSV，只包含命中模式的股票。
`mystock pattern --plot-all` 会额外为所有命中股票生成图形，输出到 `reports/plots/YYYY-MM-DD/`。

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

## 代理

当前默认已在 [default.yaml](C:\Users\wdyab\Desktop\wdy\stocks\config\default.yaml) 中配置：

```yaml
network:
  http_proxy: http://127.0.0.1:7897
  https_proxy: http://127.0.0.1:7897
  no_proxy: 127.0.0.1,localhost
```

如果你后面更换代理端口，只需要修改这三个值，不需要改 Python 代码。
