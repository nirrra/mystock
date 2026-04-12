# Project Stock Picker 自动选股与摘要增强设计

## 1. 背景与目标

当前项目已经具备以下能力：

- `mystock pattern` 生成包含模式识别结果的 CSV
- `mystock tradingview` 生成最近 5 个交易日的 TradingView 技术评分
- `mystock divergence` 将最近 15 个交易日内的 MACD 顶背离、底背离信息并入模式结果
- `daily_screening` 串行执行筛选流程，并把候选股写入 [`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md)
- `skills/project-stock-picker` 基于最新 `patterns_all_*.csv` 输出稳定型候选股 JSON

用户这次提出的需求不是新增一套完全独立的选股系统，而是对现有 `project-stock-picker` 与 `daily_screening` 链路做一次收敛性增强，目标包括：

- 适配 `patterns` CSV 新目录 `reports/patterns/`
- 保留并展示 MACD 顶背离、底背离，但不把其纳入打分
- 将主题参考文件从单一博主的 `老马-主线.md` 切换为综合整理后的 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md)
- 在选股表格后自动生成结构化摘要，包括市场情绪、主线变化、选股变化和值得注意的个股
- 明确排除“真正的指数”进入候选股结果

这次改动的重点不是改变模式逻辑本身，而是让“候选股筛选 + 每日输出说明”更符合日常盘前/盘后复盘使用。

## 2. 问题定义

当前实现存在四个明显问题：

### 2.1 `patterns` 默认目录已迁移，但 helper 仍在读旧位置

`mystock pattern` 现在默认把 CSV 写入 `reports/patterns/`，但 `skills/project-stock-picker/scripts/project_stock_picker.py` 仍然在项目根下扫描 `reports/patterns_all_*.csv`。这会导致 helper 找不到最新文件，或者误用历史旧文件。

### 2.2 MACD 背离已进入 CSV，但 helper 与摘要没有把它当成“只展示、不打分”的信息层

当前 `patterns` CSV 已包含：

- `macd_top_divergence_15d`
- `macd_bottom_divergence_15d`

这些字段应该作为风险/观察信息展示给用户，但不能把它们混进稳定型打分，否则会把“提示性信号”错误当作“择股权重”。

### 2.3 主线参考文件已从单博主切换为多博主整理，但 skill 说明仍然绑定旧文件

当前实际使用的主题判断文件已经变为 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md)，并且内容来源是多个博主与归档内容的综合整理。继续引用 `老马-主线.md` 会造成文档与实际项目状态不一致。

### 2.4 现有每日输出只有表格和一句总结，缺少稳定可复用的自动摘要

目前 [`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 主要包含分梯队表格，后面只有一段简短总结。对于日常使用，这不足以回答以下问题：

- 今天市场情绪偏强还是偏谨慎
- 当前主线有没有变化
- 与上一日相比新增了哪些票、淘汰了哪些票
- 哪些票虽然入选，但需要额外警惕顶背离或值得关注底背离

系统需要提供结构化、规则驱动、可测试的自动摘要，而不是自由发挥的长文复盘。

## 3. 目标边界

### 3.1 目标

- `project-stock-picker` 只读取 `reports/patterns/patterns_all_*.csv`
- `project-stock-picker` 输出包含候选股和结构化分析摘要的 JSON
- 保留现有稳定型打分逻辑，不把 MACD 背离并入打分
- 在最终表格中展示“符合模式/背离”
- 使用 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 作为主线判断与摘要参考文件
- 在表格后生成 4 个固定摘要段落：
  - `当日市场情绪监测`
  - `主线变动`
  - `选股变化`
  - `值得注意的股`
- 排除真正的指数进入候选股

### 3.2 非目标

- 不重写 `pattern`、`tradingview`、`divergence` 的底层计算逻辑
- 不把 `主线.md` 变成一个完整股票池映射数据库
- 不引入自由生成式的长篇复盘文案
- 不扩展到 ETF、LOF 或基金产品过滤
- 不把背离信号作为涨跌预测器或排名权重

## 4. 方案比较

### 4.1 只改 `daily_screening`

做法是保留 `project-stock-picker` 只输出候选股，由 `daily_screening` 再去额外读取 `主线.md`、历史 `选股.md` 和模式文件，自己拼接摘要。

优点：

- 改动看起来集中在一个模块

缺点：

- 选股逻辑和摘要逻辑分散
- `project-stock-picker` 作为 helper 无法独立复用
- 测试边界变差，结构化结果不易复用

结论：

不推荐。

### 4.2 `project-stock-picker` 输出结构化候选与分析，`daily_screening` 只负责渲染

做法是把当前 helper 从“只挑票”升级为“挑票 + 产出分析 JSON”，`daily_screening` 消费同一份 JSON 渲染 Markdown。

优点：

- 职责清晰
- 选股和摘要共用同一份数据源
- 结构化结果可测试、可复用
- 后续如果需要终端直接展示，也不必重写分析逻辑

缺点：

- 需要同时修改 helper、skill 文档和 `daily_screening`

结论：

这是本次推荐方案。

### 4.3 重做为大型“选股引擎”

做法是把主题识别、摘要生成、股票归类、历史比较都重写成一个更大的统一模块。

优点：

- 理论上结构最统一

缺点：

- 明显超出本次需求
- 风险过高
- 会拖慢现有日常使用流程

结论：

不采用。

## 5. 总体设计

本次采用“结构化 helper + 轻渲染 daily_screening”的设计：

`patterns CSV -> project_stock_picker.py -> JSON payload -> daily_screening Markdown`

### 5.1 输入

`project_stock_picker.py` 需要读取三类输入：

- 最新模式文件：`reports/patterns/patterns_all_YYYY-MM-DD.csv`
- 当前主线文件：[`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md)
- 历史选股文件：[`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md)

### 5.2 输出

helper 继续输出 JSON，但从当前结构：

```json
{
  "source_file": "...",
  "daily_columns": [...],
  "candidates": [...]
}
```

扩展为：

```json
{
  "source_file": "...",
  "theme_source_file": "...",
  "daily_columns": [...],
  "candidates": [...],
  "analysis": {
    "market_sentiment": {...},
    "mainline_changes": {...},
    "pick_changes": {...},
    "notable_stocks": [...]
  }
}
```

`daily_screening` 不再只生成一句“总结”，而是渲染：

- 分梯队表格
- 4 个固定摘要段落

## 6. 模块设计

### 6.1 `project_stock_picker.py`

#### 文件发现

- `_latest_patterns_file()` 改为只扫描 `reports/patterns/patterns_all_*.csv`
- 找不到文件时，错误文案同步更新为新路径

#### 候选构建

- 继续读取并标准化：
  - `symbol`
  - `name`
  - `pattern_id`
  - `tradingview_avg_all_rating_5d`
  - `tradingview_all_rating_label`
- 保留读取：
  - `macd_top_divergence_15d`
  - `macd_bottom_divergence_15d`
- `stable_score` 继续只由以下信息决定：
  - 五日平均评分
  - 最新日评分
  - pattern 优先级
  - TradingView 标签加减分
- 不把 MACD 顶背离、底背离纳入 `stable_score`

#### 排除指数

用户明确要求只排除“真正的指数”，不扩展到 ETF/LOF。

由于 helper 当前只能稳定拿到代码和名称，且项目本地 `universe.csv` 可能不存在，因此本次采用保守名称过滤：

- 仅排除名称明显为指数的行
- 关键词与白名单式规则包括但不限于：
  - `指数`
  - `上证综指`
  - `深证成指`
  - `创业板指`
  - `沪深300`
  - `中证500`
  - `中证1000`
  - `科创50`

过滤目标是“真正的指数名称”，而不是所有指数相关产品，因此不因为名称中出现行业、主题或基金术语就扩大排除范围。

#### 主题参考

- helper 文档与实际逻辑统一引用 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md)
- [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 的用途包括：
  - 提取当前主线层级
  - 为表格后的 `主线变动` 段落提供依据

### 6.2 主题映射策略

本次不把 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 强行转成“股票 -> 主题”的映射表。

个股的 `行业/主线` 继续采用防御性策略：

- 优先复用历史 [`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 中已经存在的 `symbol -> theme` 映射
- 如果没有历史映射，则保留 `未分类`
- 不因为某只股票被主线文件中顺手提及或与某赛道名称相近，就自动归类

这样做的原因是：

- [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 当前是赛道级结论，不是完整股票池
- 强行自动归类会制造大量误标

### 6.3 `daily_screening.py`

#### 表格

表格列保持不变：

- `梯队`
- `股票代码`
- `股票名称`
- `行业/主线`
- `符合模式/背离`
- `五日分数`
- `五日均分`
- `TradingView标签`
- `推荐理由`

其中 `符合模式/背离` 统一展示为：

- `pattern 1`
- `pattern 1 + 顶背离`
- `pattern 3 + 底背离`
- `pattern 2 + 顶背离 / 底背离`

#### 表格后摘要

表格后固定输出 4 段摘要：

- `当日市场情绪监测：...`
- `主线变动：...`
- `选股变化：...`
- `值得注意的股：...`

不再只输出一句“总结”，避免信息过度压缩。

## 7. 摘要生成规则

### 7.1 当日市场情绪监测

该段只使用结构化指标，不做自由发挥。推荐参考指标：

- 候选股总数
- 第一梯队数量和占比
- `strong_buy` / `buy` 数量分布
- `tradingview_avg_5d` 的均值或中位数
- 顶背离数量
- 底背离数量

输出目标是判断当日状态更接近：

- 偏强共振
- 强中有分化
- 偏谨慎观察

并给出一句解释，例如：

- 强度是否集中在少数主线
- 顶背离是否偏多
- 底背离是否提供观察机会

### 7.2 主线变动

该段由两部分组成：

- 从 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 中提取当前“核心主线 / 次主线 / 轮动线 / 短线题材”
- 对照当日候选股主题分布与上一期结果，说明哪些主线仍在延续、哪些主线今天缺席、哪些只是轮动出现

如果候选股缺乏足够主题映射，则降级为：

- 直接引用 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 的当前排序
- 明确说明“候选股主题映射不足，暂不对个股主线归属做强判断”

### 7.3 选股变化

该段对比上一期 [`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 中最近一节，输出三类集合：

- 新增：今天入选、上一期未入选
- 保留：今天和上一期都入选
- 移除：上一期入选、今天未入选

如果没有上一期历史，则固定输出：

- `首期记录，暂无上一期可比数据。`

### 7.4 值得注意的股

该段只列 2 到 5 只股票，按规则筛选，不做自由发挥。

优先级规则：

- 第一梯队且 `TradingView` 为 `strong_buy`
- 连续多日重复入选
- 出现底背离，适合作为观察项
- 出现顶背离，需要提示防追高

每只股票的说明保持短句，强调“为什么值得关注”或“为什么要多看一眼”。

## 8. 兼容与降级策略

- 若不存在上一期 [`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md)，`选股变化` 不报错，输出固定降级文案
- 若 [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 无法解析出清晰层级，则只使用“结论”段落
- 若候选股缺失 MACD 字段，则视为无背离，不中断流程
- 若个股没有历史主题映射，保留 `未分类`
- 若 helper 未能产出足够候选股，仍生成摘要，但明确指出“当日候选不足”

## 9. 测试设计

至少补充以下测试：

### 9.1 `project_stock_picker.py`

- 能正确从 `reports/patterns/` 找到最新 `patterns_all_*.csv`
- `macd_top_divergence_15d` 与 `macd_bottom_divergence_15d` 会进入输出，但不影响打分排序
- 真正的指数名称会被排除
- 普通股票不会因为名称模糊被误排除

### 9.2 `daily_screening.py`

- 表格中的 `符合模式/背离` 正确渲染
- 表格后会追加 4 段固定摘要
- 没有上一期历史时，`选股变化` 能正常降级
- 没有主题映射时，`主线变动` 能正常降级

## 10. 风险与取舍

### 10.1 风险

- [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) 是人工整理结果，不是严格结构化数据，解析规则过重会变脆
- 历史主题映射不足时，`主线变动` 的个股归因能力有限
- “值得注意的股”若规则过多，容易让解释变得不稳定

### 10.2 取舍

本次选择：

- 让摘要保持结构化、可测试，而不是追求像人工长文复盘
- 让主题映射保持保守，不因主线文件内容丰富就强行自动归类全部个股
- 让 MACD 背离承担展示与提醒职责，而不混入稳定型打分

这三个取舍能保证系统对日常使用真正有帮助，同时避免引入难以维护的“伪智能解释层”。

## 11. 实施顺序

建议按以下顺序实现：

1. 更新 `project_stock_picker.py` 的 `patterns` 路径、指数过滤和 JSON 结构
2. 更新 `SKILL.md` 中的数据源说明与输出规则
3. 扩展 `daily_screening.py`，渲染新的 4 段摘要
4. 补测试并用本地伪造样本验证
5. 再用真实项目文件跑一次 `daily_screening` 进行冒烟验证
