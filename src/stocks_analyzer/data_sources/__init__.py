from .akshare_provider import AKShareDataProvider
from .baostock_provider import BaoStockDataProvider
from .base import DataProvider


def create_data_provider(provider_name: str) -> DataProvider:
    normalized = provider_name.lower()
    if normalized == "akshare":
        return AKShareDataProvider()
    if normalized == "baostock":
        return BaoStockDataProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")


__all__ = ["AKShareDataProvider", "BaoStockDataProvider", "DataProvider", "create_data_provider"]
