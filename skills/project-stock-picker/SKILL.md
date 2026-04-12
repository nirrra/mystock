---
name: project-stock-picker
description: Use this only in the stocks project at C:\Users\wdyab\Desktop\wdy\stocks when the user asks to 选股, 挑股票, 推荐标的, 从最新 patterns 中找票, or asks for stable-profit stock candidates. Read the latest reports/patterns/patterns_all_YYYY-MM-DD.csv, use TradingView-enhanced scores, combine with 主线.md, and answer with tiered Markdown tables plus structured daily conclusions.
---

# Project Stock Picker

This skill is only for the local project at `C:\Users\wdyab\Desktop\wdy\stocks`.

## Goal

When the user asks for stock picks, do all of the following:

1. Read the latest `reports/patterns/patterns_all_YYYY-MM-DD.csv`
2. Prefer stocks whose pattern and TradingView strength are both strong
3. Use [主线.md](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) as the current market mainline reference
4. Optimize for stable gains, not maximum beta
5. Output Markdown tables by tier plus structured daily conclusions

## Workflow

1. Confirm you are in the stocks project root:
   `C:\Users\wdyab\Desktop\wdy\stocks`

2. Run the helper script:

```powershell
$env:PYTHONPATH='src'; python 'C:\Users\wdyab\Desktop\wdy\stocks\skills\project-stock-picker\scripts\project_stock_picker.py' --project-root 'C:\Users\wdyab\Desktop\wdy\stocks' --write-picks
```

3. Read [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md) and use it as the mainline reference.

4. Keep candidates that still fit the current mainline framework, especially:
   - `电池`
   - `算力硬件`
   - `创新药`
   - `机器人`

5. Treat `稳定币/支付` as lower priority by default.
   Only keep it when the user explicitly asks for short-term/event-driven trades.

6. Optimize for stable gains:
   - Prefer `pattern 1` and `pattern 3` first
   - Then consider `pattern 2`
   - Treat `pattern 4` as more aggressive and lower priority unless the user wants stronger momentum
   - Prefer `TradingView` label `strong_buy` over `buy`
   - Prefer higher `tradingview_avg_all_rating_5d`
   - Show `顶背离/底背离`, but do not use MACD divergence as a ranking weight
   - Exclude obvious true index rows if they appear in the source file
   - Avoid chasing names that look purely speculative when calmer candidates exist in the same mainline

## Output Rules

Always answer with Markdown tables.

Start with one short sentence naming the data file used.

Calling the helper in normal skill usage should also update [`选股.md`](C:/Users/wdyab/Desktop/wdy/stocks/选股.md). If the same trade date section already exists, replace that section instead of appending a duplicate.

Then output 2 to 3 tier tables. Use these columns:

- `梯队`
- `股票代码`
- `股票名称`
- `行业/主线`
- `符合模式/背离`
- `五日分数`
- `五日均分`
- `TradingView标签`
- `推荐理由`

`五日分数` means the 5 recent TradingView daily scores joined in date order, for example:
`0.28 / 0.51 / 0.45 / 0.49 / 0.52`

`行业/主线` should use the project-level theme labels, not strict申万行业. Use labels like:

- `电池`
- `算力硬件`
- `创新药`
- `机器人`
- `稳定币/支付`

If a stock cannot be matched to a mainline with reasonable confidence, keep it as `未分类` unless you have solid project-local evidence to classify it.

After the tables, always add these four lines:

- `当日市场情绪监测：...`
- `主线变动：...`
- `选股变化：...`
- `值得注意的股：...`

## Guardrails

- Do not browse the web for this task unless the user explicitly asks for latest external validation.
- Do not use stale pattern files when a newer `reports/patterns/patterns_all_*.csv` exists.
- Do not recommend solely by TradingView score without checking the pattern.
- Do not recommend solely by theme without checking the pattern and TradingView score.
- Do not turn MACD divergence into a score bonus.
- If no suitable candidates remain after mainline filtering, say so directly and still provide the four structured conclusion lines.

## Notes

- The helper script already finds the newest `reports/patterns/patterns_all_*.csv`, normalizes symbols, excludes obvious true indexes, computes a stability-oriented score, and assigns base tiers.
- Your final answer should still apply judgment after reading [`主线.md`](C:/Users/wdyab/Desktop/wdy/stocks/主线.md).
- For stable-profit requests, the final ordering should lean toward smoother trend continuation over the most explosive names.
