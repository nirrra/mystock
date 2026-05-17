# A Share Analyzer

这个项目用于辅助每日 A 股选股。当前主线是：**先判断市场主线，再从主线板块里选个股**。盘后生成板块候选池 `watchlist_sectors`、股票候选池 `watchlist_stocks` 和次日盘中源池 `intraday_pool`；盘中普通模式读取前一个交易日的 `intraday_pool`，如果当天跑过全市场盘中刷新，则优先读取当天新生成的 `intraday_pool`。

注意：最终选股文字会使用 ChatGPT 协助整理，只能作为候选参考，不是自动交易信号，也不能替代自己的买卖决策。

想看模型、回测、实现细节和历史方案，请看 [项目详细说明](docs/项目详细说明.md)。

## 快速入口

| 想看什么 | 入口 |
|---|---|
| 主线记录 | [主线.md](主线.md) |
| 盘后主线跟踪 | [reports/sectors/sector_mainline_daily_tracking.xlsx](reports/sectors/sector_mainline_daily_tracking.xlsx) |
| 盘后板块候选池 | [reports/watchlists/watchlist_sectors_YYYY-MM-DD.csv](reports/watchlists/) |
| 盘中主线跟踪 | [reports/sectors/sector_mainline_intraday_tracking.xlsx](reports/sectors/sector_mainline_intraday_tracking.xlsx) |
| 盘中板块强度 | [reports/intraday_screening/intraday_sector_strength_YYYY-MM-DD.csv](reports/intraday_screening/) |
| 盘后选股 | [选股.md](选股.md) |
| 盘中选股 | [选股-日中.md](选股-日中.md) |
| 盘后股票候选池 | [reports/watchlists/watchlist_stocks_YYYY-MM-DD.csv](reports/watchlists/) |
| 盘中股票候选池 | [reports/intraday_screening/intraday_pool_screening_YYYY-MM-DD.csv](reports/intraday_screening/) |
| 全市场Pattern结果 | [reports/patterns/patterns_all_YYYY-MM-DD.csv](reports/patterns/) |

## 每日建议动作

盘中 10:00：

```powershell
$DATE = "2026-05-11"
python -m stocks_analyzer --project-root . intraday-screening --date $DATE
```

盘中 11:40：

```powershell
$DATE = "2026-05-11"
python -m stocks_analyzer --project-root . intraday-screening --date $DATE --refresh-full-market-pool
```

运行后按 [intraday-picks-writing-guide.md](docs/intraday-picks-writing-guide.md) 更新 [选股-日中.md](选股-日中.md)。

盘中 14:30：

```powershell
$DATE = "2026-05-11"
python -m stocks_analyzer --project-root . intraday-screening --date $DATE
```

运行后按 [intraday-picks-writing-guide.md](docs/intraday-picks-writing-guide.md) 更新 [选股-日中.md](选股-日中.md)。

如果今天已经成功拉过盘中行情，只想重算筛选：

```powershell
python -m stocks_analyzer --project-root . intraday-screening --date $DATE --skip-intraday-update
```

盘后 17:30：

```powershell
$DATE = "2026-05-11"
python -m stocks_analyzer --project-root . daily-screening --date $DATE
```

运行后按 [picks-writing-guide.md](docs/picks-writing-guide.md) 更新 [选股.md](选股.md)。如果有参考博主或外部观点，把材料放入 [reports/xueqiu](reports/xueqiu/) 或相应归档目录，先更新 [主线.md](主线.md)，再整理选股。

## 接口受限

盘中默认接口是新浪 `sina_raw`。如果需要临时改用东财：

```powershell
python -m stocks_analyzer --project-root . intraday-screening --date $DATE --data-interface eastmoney_direct
```

如果接口批量受限，可以降低批量大小：

```powershell
python -m stocks_analyzer --project-root . intraday-screening --date $DATE --chunk-size 10
```

盘后日线默认走 `sina`。如日线更新失败，可先单独换接口更新：

```powershell
python -m stocks_analyzer --project-root . update --start-date 20240101 --end-date 20260511 --data-interface eastmoney
python -m stocks_analyzer --project-root . update --start-date 20240101 --end-date 20260511 --data-interface baostock
```

行业和概念映射使用同花顺 F10 个股页反查。默认只在本地映射文件缺失、为空或修改时间超过 7 天时联网刷新；每天仍会基于本地映射重新生成当日行业/概念表现。

```powershell
python -m stocks_analyzer --project-root . update-sector-membership --date $DATE
python -m stocks_analyzer --project-root . update-sector-membership --date $DATE --force-refresh
```

## 主线 + 个股选股逻辑

每日复盘先看主线，不先看个股分数。

1. **找长期主线**：看 `sector_mainline_daily_tracking.xlsx` 和 `watchlist_sectors` 的长期主线指数，找长期受资金追捧、能反复炒作的板块或概念。这里要结合 [主线.md](主线.md) 中博主观点，参考“三三制”里适合反复轮动的方向。
2. **找近期主线**：在长期主线中，看短期主线指数、最近涨幅、上涨家数和 P9，找最近开始带动市场、或处在主升初期的板块。
3. **看当日强度**：盘后看板块当日表现，盘中看 `intraday_sector_strength` 中龙头股盘中平均涨幅，判断主线是增强、延续、减弱，还是发生切换。
4. **再选个股**：只在这些主线板块/概念里挑股票，优先选龙头指数高、P1/P2/P4 结构好、P4 五日均分稳定、并且有 pattern 或技术结构支撑的股票。

如果量化主线和博主观点一致，该方向优先级最高；如果不一致，只观察，不重仓。若长期主线、短期主线和 P9 头部强度都不高，说明市场主线混沌，应减少新开仓。

## 当前核心指标

展示用分数统一是 0-100，分数越高越适合观察或买入。

| 指标 | 含义 | 用法 |
|---|---|---|
| P1 | 尾部下跌风险过滤，越高表示短期尾部风险越低 | P1 太低说明下跌风险高；非模式票不应太低 |
| P2 | Triple-barrier / CUSUM 交易型风险，越高表示先触发下行风险的概率越低 | P2 太低说明交易型风险高 |
| P4 | Qlib Alpha158 + LightGBM 收益排序，越高表示横截面收益排序越靠前 | 当前个股排序主轴 |
| P4五日均 / std | 最近 5 个交易日 P4 均值和波动 | 均值高、std 低说明上涨排序更稳定 |
| P9 | 板块买入概率，预测板块等权指数第 20 个未来交易日收盘涨幅是否超过 5% | 只用于板块池，不直接给个股排序 |
| 长期主线指数 | 板块过去约两年的资金关注、趋势、超额和抗跌表现 | 判断是否属于长期值得跟踪的主线 |
| 短期主线指数 | 板块最近 5-20 日涨幅、当日涨幅、上涨家数和短期斜率 | 判断是否正在带动市场 |
| 龙头指数 | 个股在所属板块中过去两年的长期领涨和波段领涨能力 | 用于从板块里挑龙头 |
| ATR14 / ATR% | 14 日平均真实波幅 | 用于止损距离和最大仓位 |

P3、P5、P7、P8、P10 已废弃。新盘后/盘中流程不再调用、不展示，也不作为选股依据。

## 盘后流程

`daily-screening` 当前执行：

1. 更新全市场日线。
2. 更新行业/概念映射；缓存未超过 7 天时不重抓映射，只重算当日板块表现。
3. 计算 MACD、ATR。
4. 对全市场计算 P1、P2、P4。
5. 对全市场匹配六个 pattern。
6. 计算所有行业/概念的龙头指数。
7. 计算行业/概念 P9；没有模型则 P9 留空并提示需要先训练。
8. 生成 `watchlist_stocks_YYYY-MM-DD.json/csv`。
9. 生成 `intraday_pool_YYYY-MM-DD.json/csv`，作为下一交易日盘中普通模式的 P1/P2/P4 综合分 Top200 源池。
10. 生成 `watchlist_sectors_YYYY-MM-DD.json/csv`，并更新 `sector_mainline_daily_tracking.xlsx` 的长线主线、短线主线、主线买入分三张表。生成时会补算并缓存 `sector_mainline_scores_YYYY-MM-DD.csv`，用于长期主线指数和短期主线指数。

龙头指数会同时保存两个文件：`sector_leaders_YYYY-MM-DD.csv` 用于查看每个板块的 Top 龙头，`sector_leader_scores_all_YYYY-MM-DD.csv` 保存所有有效股票在所属板块内的龙头指数，参数查询工具优先读取后者。

股票池合并两类来源：

1. `pattern`：命中任意模式的股票直接入池，只在最终统一排除当日涨幅 `> 9.9%`。
2. `phase4_top`：先排除 P1/P2 各自最高风险 20%，再要求 P1/P2/P4 达到现有门槛，并按 `P4 + P1/P2 接近 80 的加分` 取 Top20。

板块池使用排序名额制：

1. 按 `长期主线指数` 取 Top100，作为 `watchlist_sectors` 主池。
2. 在长期主线 Top100 内部，`短期主线指数` Top10 标记为短期主线。
3. 在长期主线 Top100 内部，`P9买入分` Top10 标记为 P9 高买入分；P9 模型缺失时该项留空，不影响长期主线输出。

主线判断不能只看数量。若 Top100 里头部长期主线指数、短期主线指数和 P9 分数都不强，说明市场可能处于主线混沌状态，此时应减少新开仓，只观察少数高分共振方向。

每个入选板块列出龙头股前三名。写 [选股.md](选股.md) 时，先看 `watchlist_sectors` 判断当天主线，再从 `watchlist_stocks` 里挑与长期主线中短期走强或 P9 高分板块匹配的股票；也可以直接看这些板块的龙头股。

## 盘中流程

`intraday-screening` 普通模式默认读取前一个交易日的 `intraday_pool`，即盘后生成的 P1/P2/P4 综合分 Top200 源池，并合并 `track_stock.xlsx` 中的手动跟踪股。如果当天已经用 `--refresh-full-market-pool` 跑过全市场模式，则优先读取当天的 `intraday_pool`。它用盘中临时日 K 重新计算 P1/P2/P4、MACD、ATR 和建议仓位，但不写入正式日线。

盘中选股应先看前一日 `watchlist_sectors` 的板块和龙头。当日板块强度可以用这些龙头股的盘中涨幅均值估计，并会更新到 `sector_mainline_intraday_tracking.xlsx`，再结合候选股自己的 P1/P2/P4 和涨幅判断是否适合低吸或观察。

`track_stock.xlsx` 固定三张表：`Sheet1` 只维护跟踪股编号；盘后 `daily-screening` 自动把跟踪股盘后数据写入 `Sheet2`；盘中 `intraday-screening` 自动把跟踪股盘中数据写入 `Sheet3`。

## 六个 pattern

pattern 只是候选结构，不是独立买点。

| 模式 | 名称 | 识别重点 |
|---|---|---|
| 模式1 | 量顶天立地预突破 | 长时间消化老前高后，价格接近关键前高但还未有效突破 |
| 模式2 | 量顶天立地突破确认 | 放量突破老前高，关注突破位能否站稳 |
| 模式3 | 量顶天立地突破后缩量回踩 | 突破后回踩，要求缩量且不有效跌破关键承接位 |
| 模式4 | 老鸭头鸭鼻孔金叉 | 鸭头顶后缩量回调，回调低点后 MA5 再上穿 MA10 |
| 模式5 | 趋势回踩 | 上升趋势中回踩 MA20 或趋势支撑后尝试修复 |
| 模式6 | 倍量阳支撑线反抽 | 倍量阳形成支撑线，回落到支撑附近后缩量企稳反抽 |

## 推荐交易策略

单笔交易最大亏损按账户资金 `E` 的 `2%` 控制，单一标的总仓位上限 `40% E`。

设：

```text
P = 计划开仓价
ATR = ATR14
D = 2 * ATR
```

因为计划分批买入，第二批在回撤 `D/2` 附近买入，第三次加仓前止损仍在 `P - D`，前两批有效平均止损距离按 `0.85D` 估计：

```text
理论总仓位比例 = 0.02 * P / (0.85 * D)
              = 0.02 * P / (0.85 * 2 * ATR)
              ≈ 0.01176 * P / ATR
```

最终仓位：

```text
建议总仓位 = min(40%, 理论总仓位比例)
```

四批买入：

| 批次 | 仓位 | 位置 |
|---|---:|---|
| 第一批 | 30% | 开仓点 P |
| 第二批 | 30% | 回踩结构线或回撤约 `D/2` 后确认 |
| 第三批 | 20% | 上涨到 `1R` 后回踩不破 |
| 第四批 | 20% | 创新高且趋势延续 |

止盈止损：

| 阶段 | 规则 |
|---|---|
| 开始 | 初始止损为 `P - D` |
| 上涨到 `1R` | 止损上移到 `P` 附近 |
| 上涨到 `1.5R` | 卖出 20%-30% |
| 上涨到 `3R` | 再卖出 20%-30% |
| 剩余仓位 | 按最高点 `- 2.5 * 实时 ATR` 动态止损 |
| 跌破移动止损 | 清仓，不临时下移止损 |
| 时间止损 | 买入后 5 个交易日仍不能脱离成本区或不能站稳结构位，主动降仓或退出 |

回测显示，一次性买入情况下，固定持有 60 日的平均收益最高。但实际交易不宜机械死扛，因此执行上采用 ATR 仓位、分批买入、分批止盈和动态止损。
