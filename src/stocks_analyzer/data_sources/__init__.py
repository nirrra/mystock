from .akshare_provider import AKShareDataProvider
from .baostock_provider import BaoStockDataProvider
from .base import DataProvider
from .itick_provider import ITickDataProvider
from .tushare_provider import TushareDataProvider


def create_data_provider(provider_name: str) -> DataProvider:
    normalized = provider_name.lower()
    if normalized == "akshare":
        return AKShareDataProvider()
    if normalized == "baostock":
        return BaoStockDataProvider()
    if normalized == "itick":
        return ITickDataProvider()
    if normalized == "tushare":
        return TushareDataProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")

__all__ = [
    "AKShareDataProvider",
    "BaoStockDataProvider",
    "ITickDataProvider",
    "TushareDataProvider",
    "DataProvider",
    "create_data_provider",
]
