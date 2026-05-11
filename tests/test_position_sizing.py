from __future__ import annotations

from stocks_analyzer.position_sizing import recommended_position_percent_from_mapping


def test_position_sizing_treats_exported_atr_percent_as_percent() -> None:
    assert recommended_position_percent_from_mapping({"ATR%": 8.0}) == 14.71


def test_position_sizing_uses_price_and_atr_to_disambiguate_low_exported_percent() -> None:
    assert recommended_position_percent_from_mapping({"收盘价": 11.3, "ATR14": 0.1564, "ATR%": 1.38}) == 40.0


def test_position_sizing_keeps_internal_atr_ratio_unchanged() -> None:
    assert recommended_position_percent_from_mapping({"atr_pct_14": 0.08}) == 14.71
