import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt import expected_returns


def get_optimal_allocation(date, prices, total_portfolio_value, if_hrp=False):
    """
    Calculate the optimal allocation based on the given parameters.

    :param date: Date to calculate the optimal allocation
    :param prices: DataFrame of historical prices
    :param total_portfolio_value: Current total portfolio value
    :return: Dictionary of allocations and leftover cash
    """
    # Ensure the index is in the correct format
    prices.index = pd.to_datetime(prices.index)

    # Filter the prices based on the date range
    prices = prices.loc[:date]

    if if_hrp:
        mu = expected_returns.returns_from_prices(prices)

        hrp = HRPOpt(mu)

        hrp.optimize()

        cleaned_weights = hrp.clean_weights()

        # Get portfolio performance metrics
        expected_annual_return, annual_volatility, sharpe_ratio = hrp.portfolio_performance()
    else:
        # Calculate expected returns and sample covariance matrix
        mu = ema_historical_return(prices, frequency=252)

        S = CovarianceShrinkage(prices).ledoit_wolf()

        # Optimize the portfolio
        ef = EfficientFrontier(mu, S)

        # Add L2 regularization to the optimization problem
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)

        weights = ef.max_sharpe()
        # weights = ef.min_volatility()  # Uncomment this line if you prefer min volatility

        # Get the discrete allocation of each asset
        cleaned_weights = ef.clean_weights()

        # Get portfolio performance metrics
        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()

    # Get the latest prices
    latest_prices = get_latest_prices(prices)

    # Calculate the discrete allocation
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()

    # Fill the tickers that are not in the allocation with 0
    for ticker in tickers:
        if ticker not in allocation:
            allocation[ticker] = 0

    return {
        'allocation': allocation,
        'leftover': leftover,
        'weights': cleaned_weights,
        'expected_annual_return': expected_annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
    }

# Get portfolio value


# Example usage:
if __name__ == "__main__":
    # Define the tickers and date range
    tickers = ["CGL.TO", "XUS.TO", "XGB.TO", "RY.TO", "WMT", "JNJ", "AAPL", "V", "TSLA", "BRK-B", "NVDA", "BLK", "CIS",
               "DIS", "AMD", "GOOG", "AMZN", "MSFT"]
    start_date = "2017-01-01"
    end_date = "2023-08-04"  # Today's date or last available date
    # Fetch data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)
    # Extract the 'Close' prices
    prices = data['Close']

    # Add all missing dates
    all_dates = pd.date_range(start_date, end_date)
    prices = prices.reindex(all_dates)

    # Fill NaN value with previous date value
    prices = prices.fillna(method='ffill')

    # Fetch data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)

    # Extract the 'Close' prices
    prices = data['Close']

    # Get the optimal allocation
    result = get_optimal_allocation(end_date, prices, total_portfolio_value=100000, if_hrp=True)

    print("Optimal Allocation:")
    print(result['allocation'])
    print(f"\nLeftover: ${result['leftover']:.2f}")
    print("\nWeights:")
    print(result['weights'])
    print(f"\nExpected Annual Return: {result['expected_annual_return']:.2%}")
    print(f"Annual Volatility: {result['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
