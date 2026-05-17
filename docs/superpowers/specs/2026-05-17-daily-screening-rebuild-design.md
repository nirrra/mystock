# Daily Screening 重构设计

日期：2026-05-17

## 1. 目标

重做当前 `daily-screening` 和 `intraday-screening` 主流程。新系统不再围绕单一股票候选池展开，而是同时生成：

1. 股票候选池：`watchlist_stocks`
2. 板块候选池：`watchlist_sectors`

盘后先更新全市场数据、计算全市场股票模型和技术指标，再生成股票候选与板块候选。盘中则基于前一日股票候选和板块候选，只刷新相关股票与板块龙头的临时行情，用于快速判断当日主线和个股可买性。

本次迁移方式选择直接替换现有命令：

```text
daily-screening
intraday-screening
```

不新增 v2 命令，不保留旧主流程作为默认路径。

## 2. 非目标

初版不做以下事情：

- 不自动下单。
- 不把板块龙头指数作为股票入池硬条件。
- 不要求入选板块的龙头股自身 P1/P2/P4、pattern 或技术面合格；板块侧只负责识别主线、短期强度、P9 买入分和龙头名单。
- 不用分钟线训练新模型。
- 不重训 P1/P2/P4/P9。
- 不改变已有 P1/P2/P4 的模型定义。
- P3、P5、P7、P8、P10 全部废弃，不在新盘后/盘中主流程、watchlist、用户 CSV 或选股文档中展示。

## 3. 盘后流程

`daily-screening` 改为以下顺序。

### 3.1 更新全市场信息

首先执行全市场日线更新，使用现有 `update` 逻辑和默认数据源配置。

输出仍写入：

```text
data/daily/
```

### 3.2 计算全市场技术指标和股票模型分数

对全市场股票计算：

- MACD 和背离状态
- ATR14、ATR%、建议总仓位
- P1 尾部风险分
- P2 barrier 风险分
- P4 收益排序分

随后生成全市场股票评分中间表，供后续 `watchlist_stocks` 使用。面向用户的 CSV 只保留百分制分数：

- `P1分`
- `P2分`
- `P4分`
- `P1/P2/P4综合分`

不在用户 CSV 中输出 P1/P2/P4 原始模型分。

### 3.3 全市场 pattern 匹配

对全市场匹配 1-6 号 pattern。pattern 不再单独决定最终选股，只作为 `watchlist_stocks` 的来源之一和人工解释字段。

### 3.4 生成 `watchlist_stocks`

`watchlist_stocks` 由两类来源合并：

#### 来源一：Pattern 命中股

入选规则：

```text
命中任一 pattern
```

不做 P1/P2/P4 筛查，不要求 P4 大于 70。

但统一执行最终排除：

```text
当日涨幅 > 9.9% 的股票排除
```

#### 来源二：P1/P2/P4 综合排序 Top20

先执行硬过滤：

```text
排除 P1 风险最高 20%
排除 P2 风险最高 20%
排除 当日涨幅 > 9.9%
```

再保留现有 P1/P2/P4 分数门槛：

```text
P1 >= 40
P2 >= 50
P4 >= 70
```

排序使用当前 centered risk 逻辑：

```text
综合分 =
P4
+ 0.08 * max(0, 100 - 2 * abs(P1 - 80))
+ 0.12 * max(0, 100 - 2 * abs(P2 - 80))
```

取综合分 Top20。

### 3.5 检查并更新板块成分映射

检查：

```text
data/sector_membership/stock_sector_membership.csv
```

若文件不存在、为空，或文件修改时间距离当前超过 7 天，则重新抓取同花顺 F10 行业/概念映射。否则复用本地缓存。

无论是否刷新映射，每日都要基于本地映射和本地日线重新计算当日行业/概念表现。

### 3.6 计算所有股票的龙头指数

基于当前板块成分和过去两年日线，计算每只股票在所属板块中的龙头属性。沿用 `analyze-sector-leaders` 的双榜单思想：

- 长期绝对龙头
- 板块波段领涨龙头

对股票侧输出一个面向展示的龙头指数：

```text
龙头指数 = combined_leader_score
```

同时保留标签：

- 长期核心
- 波段先锋
- 双重龙头
- 长期核心候选
- 波段活跃候选

### 3.7 筛查长期主线板块

根据长期主线指数筛出关注板块列表。长期主线指数用于判断：

```text
一个板块是否长期受到资金关注，是否具备长期主线或长期产业趋势属性。
```

长期主线指数关注过去两年表现，包括：

- 两年累计收益
- 两年相对全市场超额收益
- 跑赢市场天数比例
- 长期成交活跃程度
- 长期趋势强度
- 回撤后的修复能力

输出字段使用中文：

```text
长期主线指数
```

### 3.8 标记短期主线和 P9 高分板块

在长期主线关注板块列表中继续标记：

1. 短期主线板块：代表市场当前关注度。
2. P9 高分板块：代表未来 20 个交易日潜在发力概率。

当前使用排序名额制，不使用固定阈值：

- 按 `长期主线指数` 取 Top100：纳入 `watchlist_sectors`。
- 在长期主线 Top100 内，按 `短期主线指数` 取 Top10：标记为短期主线。
- 在长期主线 Top100 内，按 `P9买入分` 取 Top10：标记为 P9 高分。

短期主线和 P9 高分不单独把非长期主线板块拉入 `watchlist_sectors`。

主线判断不能只看入选数量，还要看强度分数。如果长期主线 Top100 的头部长期主线指数、短期主线指数和 P9 分数都不强，应在选股说明中判断为市场主线偏混沌，减少新开仓，只观察少数高分共振方向。

短期主线指数关注近 5/10/20 个交易日，包括：

- 近 5 日涨幅
- 近 10 日涨幅
- 近 20 日涨幅
- 近 5/10/20 日相对全市场超额
- 上涨家数占比
- 成交额放大

P9 买入分数沿用现有定义：

```text
预测板块等权指数第 20 个未来交易日收盘涨幅 >= 5% 的概率分。
```

板块入选逻辑只看板块层指标，不看龙头股自身 P1/P2/P4、pattern 或技术面是否合格。

### 3.9 生成 `watchlist_sectors`

`watchlist_sectors` 保存入选板块及其龙头股。

每个入选板块至少记录：

- 板块名称
- 板块类型：行业 / 概念
- 当日板块涨幅
- 成交额加权涨幅
- 上涨家数
- 上涨家数占比
- 长期主线指数
- 短期主线指数
- P9 买入分数
- 板块状态：长期主线 / 短期主线 / P9高分 / 长短共振
- 龙头股 Top3

每个板块的龙头股 Top3 来自板块龙头分析结果。优先顺序：

1. 双重龙头
2. 长期绝对龙头分高
3. 波段领涨龙头分高

龙头股展示只说明“该板块历史龙头是谁”，不要求这些龙头股当日已经满足 P1/P2/P4 或技术买点。

### 3.10 调用盘后选股指南

调用或遵循：

```text
docs/picks-writing-guide.md
```

生成或更新：

```text
选股.md
```

写作逻辑调整为：

1. 先根据 `watchlist_sectors` 指出当日主线及强度。
2. 区分持续主线、短期主线、P9 潜在发力方向。
3. 再从 `watchlist_stocks` 中选择和 P9 高分板块 / 主线板块相符合的股票。
4. 额外关注 P9 高分板块的龙头股。
5. 最理想情况是：

```text
某板块既是长期主线，又是短期主线，P9 买入分仍高；
候选股票同时来自该板块，并且具有较好的 P1/P2/P4 综合分、pattern 或技术信号。
```

这里的股票质量判断发生在选股阶段，不发生在板块入选阶段。

## 4. 盘中流程

`intraday-screening` 改为以下顺序。

### 4.1 读取前一日盘后结果

读取最近一个交易日的：

```text
watchlist_stocks_YYYY-MM-DD.json
watchlist_sectors_YYYY-MM-DD.json
```

盘中不再默认扫描全市场。盘中只刷新：

- 前一日 `watchlist_stocks` 中的股票
- 前一日 `watchlist_sectors` 中每个板块的龙头股
- 手动跟踪股，如 `track_stock.xlsx`

### 4.2 更新相关股票盘中行情

更新上述股票的盘中临时日 K，写入：

```text
data/intraday/
```

### 4.3 盘中重新计算股票分数

基于盘中临时日 K 重新计算：

- P1
- P2
- P4
- P1/P2/P4 综合分
- MACD 辅助字段
- ATR 辅助字段

盘中结果只用于临时判断，不覆盖盘后正式模型文件。

### 4.4 计算板块盘中强度

对 `watchlist_sectors` 中每个板块，使用其龙头股盘中涨幅估算板块强度：

```text
板块盘中涨幅 = mean(该板块龙头股盘中涨幅)
```

同时记录：

- 龙头上涨家数
- 龙头上涨家数占比
- 龙头最高涨幅
- 龙头平均涨幅

该指标用于盘中判断哪个板块正在引领市场。

### 4.5 生成盘中结果

输出：

```text
reports/intraday_screening/watchlist_stocks_intraday_YYYY-MM-DD.csv
reports/intraday_screening/watchlist_sectors_intraday_YYYY-MM-DD.csv
```

盘中股票表只包含前一日候选股、板块龙头股和跟踪股，不代表全市场 TopN。

### 4.6 调用日中选股指南

调用或遵循：

```text
docs/intraday-picks-writing-guide.md
```

生成或更新：

```text
选股-日中.md
```

写作逻辑调整为：

1. 先指出当日各板块强度。
2. 标明哪些板块正在引领市场，哪些板块适合观察或入场。
3. 再给出盘中股票列表。
4. 个股选择优先结合：
   - 所属板块是否在盘中走强
   - 是否属于 P9 高分板块
   - P1/P2/P4 综合分是否高
   - 是否是该板块历史龙头
   - 是否出现可解释技术信号

## 5. 输出文件

### 5.1 盘后核心文件

```text
reports/watchlists/watchlist_stocks_YYYY-MM-DD.json
reports/watchlists/watchlist_stocks_YYYY-MM-DD.csv
reports/watchlists/watchlist_sectors_YYYY-MM-DD.json
reports/watchlists/watchlist_sectors_YYYY-MM-DD.csv
```

旧版 `reports/watchlists/watchlist_YYYY-MM-DD.*` 和 `watchlist_pattern_YYYY-MM-DD.*` 已废弃；`pattern` 命令只写 `reports/patterns/patterns_all_YYYY-MM-DD.csv`，不再生成旧 watchlist。

### 5.2 盘中核心文件

```text
reports/intraday_screening/watchlist_stocks_intraday_YYYY-MM-DD.csv
reports/intraday_screening/watchlist_sectors_intraday_YYYY-MM-DD.csv
```

### 5.3 中间结果

可保留模型和技术指标中间文件，但面向用户的主要入口应是：

- `watchlist_stocks`
- `watchlist_sectors`
- `选股.md`
- `选股-日中.md`

## 6. CSV 中文列名规范

所有面向用户查看的 CSV 必须使用中文列名，除非是市场通用缩写，例如：

- MACD
- ATR
- P1
- P2
- P4
- P9

### 6.1 股票 CSV 前置列

`watchlist_stocks` 前置列顺序：

1. 交易日期
2. 编号
3. 名称
4. 涨幅%
5. 来源
6. 所属行业
7. 所属概念
8. P1分
9. P2分
10. P4分
11. P1/P2/P4综合分
12. Pattern命中
13. ATR%
14. 建议总仓位%
15. 龙头指数
16. 龙头标签
17. 板块涨幅%
18. MACD状态

后续列再放：

- P4五日均分/std
- pattern 细节
- MACD 细节
- ATR 细节
- 模型版本
- 来源文件

### 6.2 板块 CSV 前置列

`watchlist_sectors` 前置列顺序：

1. 交易日期
2. 板块名称
3. 板块类型
4. 当日板块涨幅%
5. 成交额加权涨幅%
6. 上涨家数
7. 上涨家数占比
8. 长期主线指数
9. 短期主线指数
10. P9买入分
11. 板块状态
12. 龙头股Top3
13. 双重龙头

后续列再放：

- 板块成分数
- 有效成分数
- 龙头分细节
- P9 模型信息
- 成分映射更新时间

### 6.3 不在用户 CSV 中显示的字段

以下字段只保留在内部或 JSON 中，不放入用户 CSV：

- P1 原始模型分
- P2 原始模型分
- P4 原始模型分
- P9 原始概率
- 模型内部特征
- 临时排序辅助列

## 7. JSON 结构

JSON 保留更完整的信息，便于后续工具读取。

### 7.1 `watchlist_stocks`

```json
{
  "trade_date": "YYYY-MM-DD",
  "selection_policy": {},
  "source_files": {},
  "candidate_count": 0,
  "candidates": []
}
```

### 7.2 `watchlist_sectors`

```json
{
  "trade_date": "YYYY-MM-DD",
  "selection_policy": {},
  "source_files": {},
  "sector_count": 0,
  "sectors": []
}
```

## 8. 错误处理

1. P9 模型或预测文件不存在：`watchlist_sectors` 仍输出可计算的板块表现和龙头信息，P9 显示为空，并在元数据中记录。
2. 板块成分映射刷新失败：若本地缓存存在，则继续使用本地缓存；若缓存不存在，则跳过板块流程并明确报错。
3. 板块龙头计算失败：股票 watchlist 仍应生成；板块 watchlist 报错中止或输出空表，由实现阶段决定，但错误必须清晰。
4. 盘中行情接口失败：保留前一日盘后数据，盘中表中标注“无盘中信息”。

## 9. 测试要求

实现时至少补充以下测试：

1. `daily-screening` 新流程的阶段顺序测试。
2. `watchlist_stocks` 入选规则测试：
   - pattern 命中不做 P1/P2/P4 筛查。
   - P1/P2/P4 Top20 执行硬过滤和门槛。
   - 所有来源都排除涨幅 `> 9.9%`。
3. `watchlist_sectors` 入选规则测试：
   - 长期主线筛选。
   - 短期主线筛选。
   - P9 高分筛选。
   - 每板块输出龙头 Top3。
4. 中文 CSV 列名和前置列顺序测试。
5. 盘中流程只读取前一日候选和龙头股，不扫描全市场。
6. P9 缺失时的降级测试。

## 10. 实施顺序

建议分四步实现：

1. 重构输出命名和中文 CSV 层，先不改模型计算。
2. 改造 `watchlist_stocks` 规则。
3. 新增 `watchlist_sectors` 生成逻辑。
4. 改造盘中流程和两个写作指南。

每一步都保持命令可运行，避免直接一次性替换后难以定位问题。
