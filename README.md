# A Share Analyzer

面向 A 股日线数据的技术筛选、风险过滤和收益排序工具。当前主流程是：

```text
daily-screening
  -> update
  -> macd
  -> atr
  -> Phase1 tail risk
  -> Phase2 triple-barrier risk
  -> Phase4 Alpha158/Qlib return
  -> Phase7 trade-day gate
  -> Phase5 MCD crash-risk freshness check
  -> pattern 1-6
  -> phase watchlist
  -> track_stock.xlsx Sheet2
```

项目定位是“生成候选池和风险信息”，不是自动交易系统。它不会自动下单，也不会自动修改 `选股.md`。

## 最常用指令

### 每日完整筛选

盘后最常用就是这一条：

```powershell
python -m stocks_analyzer --project-root . daily-screening --date 2026-05-07
```

默认会从 `20240101` 开始做增量更新。如果首次补历史数据，建议显式给更早日期：

```powershell
python -m stocks_analyzer --project-root . daily-screening --date 2026-05-07 --start-date 20150101
```

`daily-screening` 会先判断目标日是否为交易日。不是交易日则跳过。

### 已经更新过日线后的模块指令

如果当天已经跑过 `update`，只想手工补跑中间模块，可以按下面顺序执行：

```powershell
$env:PYTHONPATH = "src"
$DATE = "2026-05-07"

python -m stocks_analyzer --project-root . macd --date $DATE
python -m stocks_analyzer --project-root . atr --date $DATE
python -m stocks_analyzer --project-root . predict-tail-risk --date $DATE
python -m stocks_analyzer --project-root . predict-barrier-risk --date $DATE
python -m stocks_analyzer --project-root . predict-alpha158-qlib-return --date $DATE
python -m stocks_analyzer --project-root . predict-trade-day-gate --date $DATE
python -m stocks_analyzer --project-root . validate-mcd-crash-risk --start-date 2015-01-01 --end-date $DATE
python -m stocks_analyzer --project-root . pattern --as-of $DATE
python -m stocks_analyzer --project-root . track-stock --date $DATE
```

注意：上面这组指令只生成中间 CSV、旧的 `watchlist_pattern` 和手动跟踪表 Sheet2，不会合成最终 Phase watchlist。最终 Phase watchlist 合成目前挂在 `daily-screening` 内部。日常仍建议直接跑 `daily-screening`，让流程一次性产出完整结果。

### 手动跟踪股票表

在项目根目录维护 `track_stock.xlsx`：

- `Sheet1` 只填写要跟踪的股票代码，不需要股票名称。第一列标题可用 `symbol`、`code`、`股票代码`、`代码` 或 `证券代码`。
- 股票代码列会被设置为文本格式，`000001` 这类代码不会丢失前导 0。
- `Sheet2` 由程序覆盖更新，不要手动改。

单独更新跟踪表：

```powershell
python -m stocks_analyzer --project-root . track-stock --date 2026-05-07
```

`daily-screening` 最后一阶段会自动执行同样的更新。`Sheet2` 使用中文表头，字段顺序是 Phase1/2/4/5/7 的 0-100 买入友好分、pattern 是否命中、MACD/ATR 技术指标。表格里不展示模型原始分数；所有 `Phase*_score_100` 都是越高越值得看，越低代表风险越大或排序越弱。

## daily-screening 做什么

### 1. 更新数据

```text
update --start-date <start> --end-date <target>
```

更新本地 `data/daily/*.parquet`。本地已有数据时按最后日期增量补齐。

### 2. 技术辅助

`macd` 生成日线 MACD、金叉死叉、顶底背离、量价背离：

```text
reports/macd/macd_YYYY-MM-DD.csv
```

`atr` 生成 ATR14、ATR%、1ATR/2ATR 止损和 2ATR/3ATR 止盈参考：

```text
reports/atr/atr_YYYY-MM-DD.csv
```

这两个模块直接读取本地日线和股票池，不再调用外部技术评分服务。

### 3. Phase1 风险模型

命令：

```text
predict-tail-risk
```

用途：全市场个股级尾部下跌风险打分。分数越高，模型认为未来短期发生尾部下跌的概率越高。

输出：

```text
reports/full_market_model/tail_risk_predictions_YYYY-MM-DD.csv
```

当前 daily-screening 会按 `phase1_risk_score` 排除最高风险 20%。

### 4. Phase2 交易型风险模型

命令：

```text
predict-barrier-risk
```

用途：基于 triple-barrier / CUSUM 事件思想，评估个股是否更容易先触发下行风险。分数越高，风险越高。

输出：

```text
reports/full_market_model/barrier_risk_predictions_YYYY-MM-DD.csv
```

当前 daily-screening 会按 `phase2_barrier_risk_score` 再排除最高风险 20%。Phase2 还会标记 `is_cusum_event`，这个字段表示是否触发 CUSUM 事件，不决定分数大小。

### 5. Phase4 收益排序模型

命令：

```text
predict-alpha158-qlib-return
```

用途：复现 Qlib Alpha158 + LightGBM 回归框架，给全市场股票输出横截面收益排序分。

输出：

```text
reports/full_market_model/alpha158_qlib_return_predictions_YYYY-MM-DD.csv
```

当前 daily-screening 在 Phase1/Phase2 硬过滤后，按 `phase4_return_score` 取 Top20 补入 watchlist。Phase4 也用于 pattern 命中股票之间的排序参考。

### 6. Phase5 极端风险画像

命令：

```text
validate-mcd-crash-risk
```

用途：按周频收益生成 MCD crash-risk 标签和传统 crash-risk 指标，作为长周期极端风险提示。

输出：

```text
reports/full_market_model/mcd_crash_annual_measures.csv
reports/full_market_model/mcd_crash_config.json
```

当前 daily-screening 不每天强制重算 Phase5。如果结果缺失，或者距离目标日超过 6 个本地交易日，会自动刷新。Phase5 只做风险提示，不做硬过滤。

### 7. Phase7 交易日闸门

命令：

```text
predict-trade-day-gate
```

用途：判断目标日收盘后，下一交易日是否适合做技术买点。它是日期级模型，不给个股排序。

输出：

```text
reports/full_market_model/trade_day_gate_prediction_YYYY-MM-DD.csv
```

字段：

- `trade_permission = allow`：允许正常观察候选。
- `trade_permission = no_trade`：当天属于最高风险 20% 的交易日，候选只作为观察池。

注意：Phase7 不会清空 watchlist。它只告诉你当天是否应该积极开新仓。

### 8. pattern 1-6

命令：

```text
pattern --as-of YYYY-MM-DD
```

用途：扫描本地日线，识别六类技术形态。pattern 是形态筛选器，不是独立买入信号。

输出：

```text
reports/patterns/patterns_all_YYYY-MM-DD.csv
reports/watchlists/watchlist_pattern_YYYY-MM-DD.json
```

pattern 阶段会补入 MACD 和 ATR 信息，不再补入旧技术评分。

### 9. Phase watchlist

daily-screening 最后会生成：

```text
reports/watchlists/watchlist_YYYY-MM-DD.json
reports/watchlists/watchlist_YYYY-MM-DD.csv
reports/daily_screening/daily_screening_YYYY-MM-DD.json
track_stock.xlsx
```

当前 watchlist 生成规则：

1. 读取 Phase1、Phase2、Phase4、Phase5、Phase7、MACD、ATR 和 patterns。
2. Phase1 排除最高风险 20%。
3. Phase2 排除最高风险 20%。
4. 在两个风险模型都通过的股票里，保留所有命中 pattern 的股票。
5. 再加入 Phase4 收益排序 Top20，去重。
6. 排序时 pattern 命中优先，其次按 Phase4 分数从高到低。
7. 每个候选记录都会带上 Phase1/2/4/5/7、pattern、MACD、ATR 信息，并同时写入 JSON 和 CSV。
8. `reports/patterns/patterns_all_YYYY-MM-DD.csv` 也会附带 Phase1/2/4/5/7 结果和 `Phase*_score_100`，方便直接查看命中 pattern 的股票。
9. 最后读取 `track_stock.xlsx` 的 `Sheet1`，覆盖更新中文表头的 `Sheet2`，方便每天查看手动跟踪股票的同一套指标。

## 六个 pattern 分别在找什么

### 模式1：量顶天立地预突破

找“长期整理后重新接近老前高，但还没有突破”的股票。

核心条件：

- 老前高必须是前后各 40 个交易日内的局部高点。
- 当前接近老前高，但仍未有效突破。
- 未突破前不能明显放量，避免前高下方放量滞涨。

定位：前高预突破潜伏，重点看次日是否放量过关键位。

### 模式2：量顶天立地突破确认

找“已经放量突破老前高，并且突破后仍站稳”的股票。

核心条件：

- 突破的是充分消化过的老前高。
- 突破日成交量创近 90 个交易日新高。
- 突破后 1 到 10 个交易日内仍在前高上方。
- 突破后没有明显跌破 MA20 容忍线，也没有短期涨得过远。

定位：突破后确认，不把突破当天本身当作唯一买点。

### 模式3：突破后缩量回踩

找“突破老前高后，短期回踩到前高或 MA20 附近，但结构尚未破坏”的股票。

核心条件：

- 近期已经完成模式2式突破。
- 当前处在突破后 1 到 10 个交易日内。
- 收盘可以回到前高下方，但仍要守住 `MA20 * 0.98`。
- 检查日要求缩量。

定位：突破后的二次上车观察位。

### 模式4：老鸭头鸭鼻孔金叉

找“第一波上涨后缩量洗盘，MA5 再次上穿 MA10”的老鸭头低吸结构。

核心条件：

- 前面有鸭颈上涨和鸭头顶。
- 鸭头顶之后缩量回调。
- 回调不有效跌破 MA60 容忍线。
- 回调低点之后，最近 8 日内 MA5 再次上穿 MA10。
- 金叉后不能重新死叉，且当前位置不能明显高出鸭头顶。

定位：鸭嘴尚未完全张开前的试错低吸点。

### 模式5：趋势回踩

找“强趋势中回踩 MA20 后重新收回”的股票。

核心条件：

- 近期有短期高点。
- MA20 和 MA60 斜率同时向上。
- 当前股价仍在 MA60 上方。
- 最近两天缩量回踩 MA20 附近，并重新站回 MA20。

定位：强趋势里的深洗盘修复。

### 模式6：倍量阳支撑线反抽

找“前面倍量阳拉升，随后缩量回踩到倍量阳收盘支撑线附近”的股票。

核心条件：

- 有一个倍量阳锚点，锚点收盘价作为支撑线。
- 锚点后出现明显拉升。
- 从峰值回落时整体缩量。
- 回踩到支撑线附近后企稳，或者跌破后快速重新站回。
- 回踩阶段最大单日量不能超过上涨末端三日均量的 1.2 倍。

定位：支撑线附近的反抽观察，分为 `support_hold` 和 `break_reclaim` 两类。

## 模型训练与验证

### 训练 daily-screening 需要的部署模型

```powershell
python -m stocks_analyzer --project-root . train-tail-risk-model --start-date 2015-01-01 --end-date 2026-05-07
python -m stocks_analyzer --project-root . train-barrier-risk-model --start-date 2015-01-01 --end-date 2026-05-07
python -m stocks_analyzer --project-root . train-alpha158-qlib-return-model --start-date 2015-01-01 --end-date 2026-05-07
python -m stocks_analyzer --project-root . train-trade-day-gate-model --start-date 2015-01-01 --end-date 2026-05-07
```

模型 artifact：

```text
data/ml/full_market_risk/tail_risk_model.pkl
data/ml/full_market_barrier_risk/barrier_risk_model.pkl
data/ml/full_market_alpha158_return/alpha158_qlib_return_model.pkl
data/ml/full_market_trade_day_gate/trade_day_gate_model.pkl
```

### 验证各阶段

```powershell
python -m stocks_analyzer --project-root . audit-full-market-data --min-exact-history-days 900 --tail-lookback-days 100 --max-horizon-days 20
python -m stocks_analyzer --project-root . validate-tail-risk-walkforward --start-date 2016-01-01 --end-date 2026-04-30 --windows 6
python -m stocks_analyzer --project-root . validate-barrier-risk-grid --start-date 2016-01-01 --end-date 2026-04-30
python -m stocks_analyzer --project-root . validate-alpha158-qlib-return --start-date 2016-01-01 --end-date 2026-04-30
python -m stocks_analyzer --project-root . validate-mcd-crash-risk --start-date 2016-01-01 --end-date 2026-04-30
python -m stocks_analyzer --project-root . validate-trade-day-gate --start-date 2016-01-01 --end-date 2026-04-30
```

### 当前 Phase 验证结论

截至 2026-05-08 已完成 Phase1/2/4/5/7 的独立验证和部署预测。结论只说明各 Phase 自身有无信息量，还不能替代最终组合消融回测。

| Phase | 当前结论 | 关键结果 | 当前用途 |
| --- | --- | --- | --- |
| Phase1 尾部风险 | 有效 | walk-forward `PR-AUC 0.151` vs 基准 `0.057`，`ROC-AUC 0.702`；高风险 decile 在所有窗口里都对应更高风险、更差回撤。过滤最高风险 20% 后，5 日未来收益略改善，最大回撤改善。 | 个股级硬风险过滤，排除最高风险 20%。 |
| Phase2 triple-barrier 风险 | 有效，交易影响强于 Phase1 | 最优网格里过滤最高风险 20% 后，5 日未来收益改善约 `0.258%`，未来最大回撤改善约 `0.40pct`；多个网格均通过风险过滤检验。 | 个股级硬风险过滤，排除最高风险 20%；保留 CUSUM event 作为事件标记。 |
| Phase4 Alpha158/Qlib 收益 | 有明显排序信号 | 测试集 `IC 0.0587`、`RankIC 0.0414`，正 IC 日比例约 `78%`；TopK 组合测试期 hit rate 约 `60%`。 | 通过 Phase1/2 后的收益排序核心，补入 Phase4 Top20。 |
| Phase5 MCD crash-risk | 风险画像合理，但不适合直接当日线硬过滤 | `NEGOUTLIER` 总体比例约 `14.8%`；与 `MINRET` 相关性约 `-0.605`，与 `SIGMA` 相关性约 `0.509`。 | 长周期极端风险提示，暂不做硬过滤。 |
| Phase7 交易日闸门 | 有效 | 最佳摘要 `PR-AUC 0.574` vs 基准 `0.446`；allow 日未来收益和最大回撤均优于整体。 | 日期级交易许可，`no_trade` 日只观察不积极开新仓。 |

当前还缺的结果：

- 新版 `pattern 1-6` 在 2016-2026 全量区间的独立回测。
- `patterns_only` 是否打赢随机。
- `patterns + Phase1/2` 是否明显优于 pure patterns。
- `patterns + Phase1/2 + Phase4` 是否优于 `Phase4 only`。
- 当前完整 `daily-screening watchlist` 是否优于各单组件基准。

因此当前判断是：模型侧 Phase1/2/4/7 已有独立价值；Phase5 暂作风险提示；pattern 和最终组合仍需要进一步消融回测确认。

### 回测 pattern

```powershell
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --save-forward-prices --forward-days 40
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --1 --save-forward-prices --forward-days 40
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --2 --save-forward-prices --forward-days 40
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --3 --save-forward-prices --forward-days 40
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --4 --save-forward-prices --forward-days 40
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --5 --save-forward-prices --forward-days 40
python -m stocks_analyzer --project-root . backtest-patterns --start-date 2016-01-01 --date 2026-04-30 --6 --save-forward-prices --forward-days 40
```

组合消融回测命令清单见：

```text
docs/superpowers/specs/2026-05-07-daily-screening-component-backtest-commands.md
```

其中统一组合回测命令 `backtest-daily-screening-components` 仍待实现。

## 目录和输出

常用输出：

```text
data/daily/                              个股日线 parquet
data/ml/full_market_risk/                Phase1 模型
data/ml/full_market_barrier_risk/        Phase2 模型
data/ml/full_market_alpha158_return/     Phase4 模型
data/ml/full_market_trade_day_gate/      Phase7 模型
reports/macd/                            MACD 和量价状态
reports/atr/                             ATR 风险辅助
reports/full_market_model/               Phase1/2/4/5/7 验证和预测
reports/patterns/                        pattern 扫描结果
reports/watchlists/                      watchlist 和 watchlist_pattern
reports/daily_screening/                 daily-screening 运行摘要
```

手工维护文件：

```text
主线.md
选股.md
docs/picks-writing-guide.md
```

## 参考项目和论文

### Phase1：尾部风险分类

- Noh, S.-H. (2026). "Predicting Stock Market Risk Using Machine Learning Classification Models." `Risks`, 14(4), 92. https://www.mdpi.com/2227-9091/14/4/92

本项目复现其“滚动 100 日 5% 分位尾部风险标签 + 多分类器比较”的思路，并扩展到 A 股全市场个股日线面板。

### Phase2：Triple-Barrier 风险标签

- Lopez de Prado, M. (2018). `Advances in Financial Machine Learning`. Wiley.
- `nkonts/barrier-method`: https://github.com/nkonts/barrier-method
- Mlfin.py labeling documentation: https://mlfinpy.readthedocs.io/en/stable/Labelling.html

本项目使用日线 open/high/low/close 构建交易型风险标签，并保留 CUSUM event 标记。

### Phase3/Phase4：Qlib Alpha158 + LightGBM

- Microsoft Qlib paper: https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/
- Qlib benchmark README: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
- Qlib LightGBM Alpha158 config: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
- Qlib Alpha158 handler source: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

Phase3 是 Alpha158 风险模型研究。Phase4 是当前 daily-screening 使用的 Alpha158/Qlib 风格收益回归模型。

### Phase5：MCD Crash-Risk

- Karasan, A., Alp, O. S., and Weber, G.-W. (2025). "Machine learning approach to stock price crash risk." `Annals of Operations Research`, 350, 1053-1074. https://link.springer.com/article/10.1007/s10479-025-06596-7

本项目只复现 crash-risk 标签和极端风险画像，不复现论文中的公司财务变量和投资者情绪回归。

### Phase7：交易日买点闸门

Phase7 是本项目基于本地 A 股全市场日线构建的市场状态模型。它使用合成指数、市场广度和技术状态判断“下一交易日是否适合技术买点”，不参考个股 pattern、Phase2 或 Phase4 作为输入。

### 后续候选：MASTER

- Li et al. (2024). "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting." AAAI 2024. https://huggingface.co/papers/2312.15235
- Official code: https://github.com/SJTU-DMTai/MASTER

MASTER 暂未进入 daily-screening。只有当 Phase4 LightGBM 回归模型在本地回测中显示稳定价值后，才考虑继续复现。

## 注意事项

- `daily-screening` 适合盘后运行，不适合盘中全市场频繁执行。
- `watchlist` 是候选池，不是最终买入名单。
- `Phase7 = no_trade` 时，watchlist 仍会生成，但应作为观察池。
- `Phase5` 是长周期极端风险画像，不是日内买点信号。
- pattern 命中只说明形态存在，不代表胜率足够高。
- `选股.md` 需要按 `docs/picks-writing-guide.md` 手工整理。
- 同一日期重复运行会覆盖同日期输出。

## 入口

推荐统一使用：

```powershell
python -m stocks_analyzer --project-root . <subcommand>
```

兼容入口：

```text
mystock
stocks-analyzer
```
