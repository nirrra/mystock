# Findings & Decisions

## Requirements

- 面向 A 股主板做技术面分析
- 不做自动交易、下单、回测撮合
- 需要日线，最好能补充分钟线
- 目标是筛出满足条件的股票，不是做交易系统
- 交付形式为命令行工具
- 关注周期为短线和波段
- 策略框架需覆盖趋势突破、回调低吸、强势股跟踪三类模板
- 数据源先免费跑通，后续保留切换付费源能力

## Research Findings

- 当前项目目录起步状态接近空仓库，仅有设计文档，无现成代码基础。
- AKShare 适合作为 A 股免费原型数据源，适合先跑通股票列表和日线流程。
- Tushare Pro 更适合后续升级为稳定或更完整的数据源，尤其在分钟线场景下更有长期可维护性，但不适合作为当前 V1 的第一依赖。
- V1 的主要技术风险不在策略表达，而在股票池定义、数据源稳定性和字段标准化。
- 对当前需求而言，日线主筛选加分钟线辅助确认是最合适的复杂度控制方式。
- 在当前网络环境下，AKShare 的东财全市场实时分页接口容易出现代理或连接中断。
- `stock_info_a_code_name()` 在当前环境下可成功返回股票代码和名称，更适合用于 V1 的股票池更新。
- `baostock` 在当前环境下可成功登录、拉取全市场证券列表，并成功抓取主板日线与 5 分钟线样例。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 项目采用分层结构：数据源、股票池、指标、筛选、策略、输出 | 降低耦合，便于替换数据源和增量扩展 |
| 数据接口统一为 `get_instruments` / `get_daily_bars` / `get_intraday_bars` | 锁定上层依赖边界 |
| V1 不引入数据库 | 文件缓存足够，先优化开发速度 |
| V1 基础指标只保留 MA、新高、均量、涨跌幅、回撤、波动率 | 用最小指标集覆盖三类模板 |
| 模板逻辑不硬编码为独立脚本 | 统一配置和筛选引擎更利于维护 |
| 根目录提供 `main.py` 作为直接入口 | 用户无需先安装包，也能先跑 CLI |
| 股票池更新改用 `stock_info_a_code_name()` | 比 `stock_zh_a_spot_em()` 更轻，当前网络下更稳定 |
| 默认 provider 切换为 `baostock` | 当前环境下 `baostock` 实测比 AKShare-东财链路更稳定 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 当前目录不是 Git 仓库，无法完成 skill 要求中的 commit | 记录为外部前置条件，不阻塞规划与文档写入 |
| `rg` 工具在当前环境拒绝访问 | 改用 PowerShell 文件探索命令 |
| `pip install -e .` 在默认沙箱中无法写用户临时目录 | 通过权限提升安装依赖 |
| `stock_zh_a_spot_em()` 真实调用失败 | 用更轻的代码列表接口替换股票池更新来源 |
| AKShare 日线接口在当前环境下持续出现代理中断 | 改为接入并默认启用 `baostock` |

## Resources

- 设计文档: `C:\Users\wdyab\Desktop\wdy\stocks\docs\superpowers\specs\2026-04-09-a-share-analysis-design.md`
- AKShare 股票数据文档: https://akshare.akfamily.xyz/data/stock/stock.html
- AKShare 数据说明: https://akshare.akfamily.xyz/data_tips.html
- Tushare 分钟数据说明: https://tushare.pro/document/1?doc_id=234
- 项目入口: `C:\Users\wdyab\Desktop\wdy\stocks\main.py`

## Visual/Browser Findings

- AKShare 官方文档显示其股票数据能力足以支撑 A 股原型阶段的日线分析流程。
- AKShare 文档同时提示部分数据接口存在字段或复权层面的使用注意事项，因此上层不能直接耦合其原始字段。
- Tushare 官方文档显示分钟数据属于更明确的专业数据能力，适合在后续升级阶段接入，而不是 V1 的主依赖。
- 官方文档显示 `stock_zh_a_hist_min_em` 的 1 分钟数据只返回近 5 个交易日且不复权，因此分钟线应保持辅助定位。
