# Task Plan: A股技术面分析框架 V1

## Goal

构建一个面向 A 股主板的 Python 命令行技术面分析框架，先基于免费数据源跑通股票池、日线缓存、基础指标、三类参数化模板和可解释结果输出。

## Current Phase

Phase 3

## Phases

### Phase 1: Requirements & Discovery

- [x] 明确用户目标为 A 股主板技术面分析，不涉及自动交易
- [x] 明确交付形式为命令行工具
- [x] 明确以日线为主、分钟线为辅
- [x] 明确三类模板：趋势突破、回调低吸、强势股跟踪
- [x] 将设计整理为正式 spec
- **Status:** complete

### Phase 2: Planning & Structure

- [x] 固定项目目录结构
- [x] 定义数据源抽象接口
- [x] 定义股票池过滤规则
- [x] 定义基础指标清单与字段规范
- [x] 定义三类模板的配置结构
- [x] 定义 CLI 命令入口
- **Status:** complete

### Phase 3: Implementation

- [x] 初始化 Python 项目结构
- [x] 实现 AKShare 数据源适配器
- [x] 实现主板股票池模块
- [x] 实现日线更新和本地缓存
- [x] 实现基础指标计算
- [x] 实现三类策略模板
- [x] 实现 CLI 命令
- **Status:** complete

### Phase 4: Testing & Verification

- [x] 为股票池过滤编写测试
- [x] 为指标计算编写测试
- [x] 为模板筛选编写测试
- [x] 跑通端到端样例
- [x] 记录验证结果和已知限制
- **Status:** complete

### Phase 5: Delivery

- [ ] 整理最终使用说明
- [ ] 说明数据源限制与后续扩展点
- [ ] 交付可运行的 V1
- **Status:** in_progress

## Key Questions

1. 如何在不依赖具体第三方字段名的情况下统一股票列表、日线和分钟线接口？
2. 主板过滤规则在免费数据源下如何稳定表达并可测试？
3. 三类模板的参数结构如何设计，才能兼顾可读性和可扩展性？

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| V1 仅做分析，不做交易执行 | 保持问题边界清晰，避免过早引入回测和下单复杂度 |
| 命令行是第一阶段唯一交付形式 | 用户已明确要 CLI，最适合快速迭代 |
| 日线是主筛选层 | 免费数据源更适合日线主流程，分钟线放在增强层更稳 |
| 分钟线仅用于候选股二次确认 | 降低数据量和接口不稳定性对系统主干的影响 |
| V1 采用 AKShare 为原型主源 | 免费、起步快，适合先跑通流程 |
| 数据源必须抽象成统一接口 | 后续切换付费源时不重写上层逻辑 |
| 本地缓存优先使用 Parquet | 实现简单、读取快、适合表格型行情数据 |
| 三类策略采用模板加参数形式 | 避免每个想法都变成一个独立脚本 |
| `intraday-screening` 的盘中排序作为独立汇总层实现 | 保持原有日线输出链路不变，降低回归风险 |
| `daily_score` 直接复用 `watchlist._stable_score()` | 避免盘中链路引入第二套日线评分口径 |
| 最终盘中结果按 `intraday_5m_score` 排序 | 让本交易日 5 分钟状态成为首要排序依据 |
| `update` 直接改为自动补全末尾缺口 | 用户明确不再需要新命令，且不允许通过更晚的 `--start-date` 制造缺口 |
| `daily-screening` 的趋势复核通过独立 `trend` 指令接入 | 复用现有趋势评分链路，避免在 `pattern/watchlist` 内重复实现一套打分 |
| `watchlist` 采用“旧体系通过 AND trend 宽松阈值通过” | 保留原有技术候选稳定性，同时叠加 `buy_score / price_action_score` 质量复核 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `rg.exe` 在当前环境启动失败 | 1 | 改用 PowerShell 原生命令探索文件 |
| 当前目录不是 Git 仓库，无法提交 spec | 1 | 用户后续已初始化 Git，阻塞已解除 |
| `pip install -e .` 在沙箱内无法写入用户临时目录 | 1 | 升级权限后完成依赖安装 |
| `stock_zh_a_spot_em()` 在当前网络环境下不稳定 | 1 | 改为 `stock_info_a_code_name()` 做股票池更新 |
| `planning-with-files` 的 catchup 脚本路径在当前机器不存在 | 1 | 直接复用仓库内已有 planning files 继续记录实现过程 |

## Notes

- 实现阶段禁止把分钟线扩张为全市场主数据层
- 优先验证“能稳定筛出合理候选股”，而不是追求指标数量
- 若用户后续要求接入付费分钟线，应先保持当前接口不变再替换底层实现
- 趋势复核第一版只影响 `watchlist` 准入，不改 `watchlist` 现有排序逻辑
- `pattern` 体系已进入六模式阶段：新 `1/2/3` 为 `量顶天立地` 三阶段，旧 `2/3/4` 已顺延为新 `4/5/6`
