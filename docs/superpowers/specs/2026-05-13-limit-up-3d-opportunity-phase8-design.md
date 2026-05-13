# Phase8 三日涨停机会模型设计

日期：2026-05-13

## 1. 目标

新增一个独立的超短期模型 Phase8，用于预测未来 3 个交易日内更可能出现短线强势机会、同时大跌风险较低的股票。

Phase8 不替代现有 Phase1/2/4，也不参与当前 watchlist 的入选和排序。第一阶段只做独立训练、预测、验证和展示。只有当严格验证证明有效后，才考虑把它接入 daily-screening 排序或人工选股优先级。

核心目标不是单纯预测“是否涨停”，而是预测：

```text
未来 3 个交易日内有触板机会，并且不出现明显三日下跌风险。
```

## 2. 非目标

Phase8 初版不做以下事情：

- 不预测分钟级涨停。
- 不预测封板强度、炸板概率或连板概率。
- 不直接改变主 watchlist 入选逻辑。
- 不作为硬过滤条件。
- 不处理 ST、科创、创业板、北交所不同涨跌停制度；初版统一使用 `> 9.9%` 的触板近似口径。

## 3. 标签定义

对每个股票、每个信号日 `t` 构造标签。特征只允许使用 `t` 日收盘后已经可见的数据，标签使用未来 `t+1` 到 `t+3` 三个交易日。

### 3.1 当日已涨停样本排除

如果信号日 `t` 当天已经触及涨停，则跳过该样本，不进入训练集：

```text
today_limit_up = high[t] / close[t-1] - 1 > 0.099
```

原因：

- 当日已经涨停的票，盘后买入可执行性差。
- 这类样本会让模型学习“已涨停后的延续”，不符合寻找买进机会的目标。
- 预测输出中仍记录 `today_limit_up_excluded`，但这类股票不进入 Phase8 TopN。

### 3.2 三日触板标签

未来 3 个交易日内，只要任一交易日触及该交易日涨停近似阈值，就记为 1：

```text
hit_3d = 1{
  max(
    high[t+1] / close[t] - 1,
    high[t+2] / close[t+1] - 1,
    high[t+3] / close[t+2] - 1
  ) > 0.099
}
```

这里用每个未来交易日自己的前一日收盘价作为基准。

### 3.3 三日大跌标签

未来 3 个交易日累计收益低于 `-5%`，记为大跌：

```text
down_3d = 1{
  close[t+3] / close[t] - 1 < -0.05
}
```

如果未来不足 3 个交易日，该样本不进入训练集。

### 3.4 冲高回落惩罚标签

如果未来 3 日内触板，但第 3 日收盘相对信号日仍大跌，视为“冲高诱多后回落”，比普通未触板样本更差：

```text
trap_3d = hit_3d & down_3d
```

### 3.5 单一训练目标

采用单一组合目标训练，不拆成两个模型：

```text
target = hit_reward * hit_3d
       - down_penalty * down_3d
       - trap_penalty * trap_3d
```

默认参数：

```text
hit_reward = 1.0
down_penalty = 1.2
trap_penalty = 1.5
```

默认取值解释：

| hit_3d | down_3d | trap_3d | target | 含义 |
|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 1.0 | 最好：三日内触板且不大跌 |
| 0 | 0 | 0 | 0.0 | 普通：没触板也没大跌 |
| 1 | 1 | 1 | -1.7 | 冲高回落：触板但最终大跌，比普通和单纯大跌更差 |
| 0 | 1 | 0 | -1.2 | 没触板且大跌 |

这个默认值按最新要求提高了防下跌权重，并让 `hit_3d=1, down_3d=1` 的冲高回落样本低于 `hit_3d=0, down_3d=1` 的单纯大跌样本。

验证阶段至少比较：

```text
down_penalty = 1.2
trap_penalty = 1.5
```

如果 TopN 中触板后大跌比例仍高，则优先提高 `trap_penalty`；如果整体大跌率偏高，则提高 `down_penalty`。

## 4. 特征和模型

Phase8 复用 Phase4 的 Alpha158 特征工程和 LightGBM 框架。

推荐模型：

```text
LightGBMRegressor
```

理由：

- 目标是横截面排序，不是绝对概率校准。
- 组合目标包含 `1 / 0 / -1` 等连续排序含义，用回归更直接。
- 现有 Phase4 训练、预测、缓存、GPU 参数和 daily-screening 快速预测模式可复用。

初版模型名：

```text
model_name = lightgbm_regressor
model_version = limit_up_3d_opportunity_phase8_v1
```

模型产物目录：

```text
data/ml/full_market_limit_up_3d/
```

## 5. 输出字段

预测文件路径：

```text
reports/full_market_model/limit_up_3d_opportunity_predictions_YYYY-MM-DD.csv
```

核心字段：

```text
trade_date
feature_trade_date
symbol
name
phase8_raw_score
phase8_score_100
phase8_rank
today_limit_up_excluded
prediction_scope
model_name
model_version
```

验证文件额外保留：

```text
hit_3d_label
down_3d_label
trap_3d_label
target
future_return_3d
future_max_high_return_3d
future_max_drawdown_3d
```

展示口径：

- `phase8_raw_score` 是模型原始输出。
- `phase8_score_100` 是当日横截面百分制分数，越高越值得短线关注。
- `phase8_rank` 是按 `phase8_raw_score` 从高到低排序。
- `today_limit_up_excluded = true` 的股票不进入 Phase8 TopN 展示。

## 6. 验证设计

Phase8 只用于看高分股，因此验证重点不是全样本误差，而是 TopN 质量。

### 6.1 Walk-forward 验证

建议沿用严格 OOS 框架：

```text
start_date = 2015-01-01
test_start_date = 2020-01-01
end_date = 2026-04-30
test_window_days = 60
step_days = 60
embargo_days = 3
min_train_days = 900
```

每个窗口只用测试期之前的数据训练，测试期内按每日横截面排序。

### 6.2 TopN 指标

统计以下组合：

```text
Top5
Top10
Top20
Top50
Top100
```

每组输出：

```text
trade_count
hit_3d_rate
down_3d_rate
trap_3d_rate
avg_target
avg_future_return_3d
median_future_return_3d
win_rate_3d
avg_future_max_high_return_3d
avg_future_max_drawdown_3d
```

重点判断：

- TopN 触板率是否高于全市场随机。
- TopN 三日大跌率是否不高于随机，最好更低。
- TopN 冲高回落率是否受 `trap_penalty` 控制。
- Top5/Top10 是否比 Top20 更适合人工短线选股。

### 6.3 对照基准

至少比较：

```text
random_topN
phase4_topN
centered_risk_topN
rolling5_phase4_mean_topN
phase8_topN
```

Phase8 只有在以下条件同时满足时才算有实用价值：

1. Top20 触板率显著高于随机和 Phase4 Top20。
2. Top20 大跌率不高于随机。
3. Top5/Top10 的触板率和平均收益有明显提升。
4. 分年度结果不是只依赖 2024-2026 牛市。

## 7. CLI 命令设计

新增训练命令：

```powershell
python -m stocks_analyzer --project-root . train-limit-up-3d-opportunity-model --start-date 2015-01-01 --end-date 2026-05-12
```

新增预测命令：

```powershell
python -m stocks_analyzer --project-root . predict-limit-up-3d-opportunity --date 2026-05-13
```

新增验证命令：

```powershell
python -m stocks_analyzer --project-root . validate-limit-up-3d-opportunity --start-date 2015-01-01 --test-start-date 2020-01-01 --end-date 2026-04-30 --top-ns 5,10,20,50,100 --down-penalty 1.2 --trap-penalty 1.5
```

建议支持参数：

```text
--limit
--latest-only
--feature-lookback-bars
--compact-output
--lgbm-device auto|cpu|gpu|cuda
--lgbm-n-jobs -1
--trap-penalty
--down-penalty
--limit-up-threshold 0.099
--down-threshold -0.05
```

## 8. daily-screening 和 intraday-screening 接入边界

第一阶段只独立展示，不参与任何筛选和排序。

daily-screening 后续可增加一个可选阶段：

```text
predict-limit-up-3d-opportunity
```

但主 watchlist 入选逻辑保持不变。

在表格中只展示：

```text
phase8_score_100
phase8_rank
today_limit_up_excluded
```

如果 `today_limit_up_excluded = true`，在人工选股中标注“当日已触板，不作为 Phase8 买进机会”。

## 9. 数据和边界条件

### 9.1 前复权与涨停判断

当前日线数据使用既有项目数据，不额外引入分钟线。触板判断用日线 high 和前一交易日 close：

```text
high / previous_close - 1 > 0.099
```

该规则是简化近似，可能把部分非主板 20% 涨幅股票也纳入“触板”。这是用户确认后的初版口径。

### 9.2 停牌和缺失

如果未来 3 个交易日不足，或未来 high/close 缺失，则样本跳过。

如果信号日缺少前一日 close，无法判断 `today_limit_up`，则样本跳过。

### 9.3 当日已触板预测过滤

训练跳过 `today_limit_up = true` 样本。预测时也要计算同一字段：

```text
today_limit_up_excluded = true
```

这类股票仍可写入 skipped 文件，或在预测主文件中保留但不参与 Phase8 TopN。

## 10. 实施模块划分

建议新增：

```text
src/stocks_analyzer/full_market_limit_up_3d.py
```

职责：

- 构造 Phase8 标签。
- 构造 Phase8 Alpha158 训练面板。
- 训练 LightGBMRegressor。
- 生成每日预测。
- 输出验证报告。

复用：

- `full_market_alpha158.build_alpha158_feature_frame`
- `full_market_alpha158.build_alpha158_latest_feature_frame`
- `full_market_return.QLIB_ALPHA158_LGBM_PARAMS`
- 现有 LightGBM device 参数处理方式
- 现有 full_market_model 报告目录

新增测试：

```text
tests/test_full_market_tail_risk.py
tests/test_cli.py
```

测试重点：

- 标签：未来任一日 high / prev_close > 9.9% 标记为 hit。
- 标签：三日 close / signal close < -5% 标记为 down。
- 标签：today_limit_up 样本被跳过。
- target：hit/down/trap 组合符合公式。
- CLI 默认参数能解析。

## 11. 通过标准

Phase8 第一阶段完成的标准：

1. 可以构造 Phase8 训练集并保存 audit。
2. 可以训练部署模型并保存 artifact。
3. 可以对指定日期生成预测 CSV。
4. 可以跑 walk-forward 验证并输出 TopN 指标。
5. 验证报告能直接判断 Phase8 TopN 是否优于 random 和全体可交易样本；是否与 Phase4/centered risk 组合比较，等 Phase8 单模型有效后再加。
6. 不改变现有 daily-screening/watchlist 排序结果。

## 12. 后续决策

如果 Phase8 验证有效，下一步再讨论：

- 是否在 watchlist 主表中展示 Phase8。
- 是否为短线用户单独生成 `short_term_watchlist_YYYY-MM-DD.csv`。
- 是否把 Phase8 高分股与 pattern 命中、主线、P1/P2 风险分做组合。
- 是否把 Phase8 用作盘中 focus 的额外参考分。

初版不做这些集成，避免在模型有效性未证明前污染现有主流程。
