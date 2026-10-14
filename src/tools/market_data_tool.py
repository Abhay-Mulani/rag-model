import csv
from pathlib import Path


def _load_stock_data(path: str = "data/raw/stock_data.csv") -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_stock_info(symbol: str, path: str = "data/raw/stock_data.csv") -> dict:
    """Return public demo metrics for a symbol from the local market dataset."""
    stock = next(
        (row for row in _load_stock_data(path) if row.get("symbol", "").upper() == symbol.upper()),
        None,
    )
    return stock or {"error": f"Stock '{symbol}' not found in database."}


def calculate_portfolio_risk(holdings: list[dict], path: str = "data/raw/stock_data.csv") -> dict:
    stocks = {row.get("symbol", "").upper(): row for row in _load_stock_data(path)}
    beta = 0.0
    sectors: dict[str, float] = {}
    total = 0.0
    missing: list[str] = []
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).upper()
        weight = float(holding.get("weight", 0))
        total += weight
        stock = stocks.get(symbol)
        if not stock:
            missing.append(symbol)
            continue
        beta += float(stock.get("beta", 1.0)) * weight
        sector = stock.get("sector", "Unknown")
        sectors[sector] = sectors.get(sector, 0.0) + weight
    maximum = max(sectors.values(), default=0.0)
    return {
        "portfolio_beta": round(beta, 3),
        "sector_weights": {key: round(value, 3) for key, value in sectors.items()},
        "max_sector_weight": round(maximum, 3),
        "sector_concentration_risk": "high" if maximum > 0.4 else "moderate" if maximum > 0.25 else "low",
        "risk_level": "high" if beta >= 1.2 else "moderate" if beta >= 0.8 else "low",
        "total_weight": round(total, 3),
        "missing_symbols": missing,
    }


def suggest_allocation(risk_appetite: str, investment_amount: float = 1000000) -> dict:
    allocations = {
        "conservative": {"large_cap_equity": 0.25, "debt": 0.4, "gold": 0.15, "cash": 0.2},
        "moderate": {"large_cap_equity": 0.4, "mid_cap_equity": 0.15, "debt": 0.25, "gold": 0.1, "cash": 0.1},
        "aggressive": {"large_cap_equity": 0.4, "mid_cap_equity": 0.25, "small_cap_equity": 0.2, "gold": 0.05, "cash": 0.1},
    }
    appetite = risk_appetite.lower()
    if appetite not in allocations:
        return {"error": "Risk appetite must be conservative, moderate, or aggressive."}
    return {
        "risk_appetite": appetite,
        "investment_amount": investment_amount,
        "allocation": {
            key: {"percentage": value, "amount": round(value * investment_amount, 2)}
            for key, value in allocations[appetite].items()
        },
    }
