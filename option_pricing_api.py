import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from volatility import volatility_trackers, parameter_estimators
from volatility.volatility_trackers import VolatilityTracker
from volatility.parameter_estimators import ParameterEstimator
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from pricing import options, curves
from pricing.curves import YieldCurve
from pricing.options import OptionType, OptionsPricer
import pandas_datareader.data as web
import pandas_market_calendars as mcal
import numpy as np
import time

yf.pdr_override()


def _get_garch(asset_prices:pd.DataFrame) -> ParameterEstimator:
    """Get for the ω, α, and β parameters of the GARCH(1, 1) model

    Args:
        asset_prices (pd.DataFrame): stock's historical prices

    Returns:
        ParameterEstimator: estimator object that contains ω, α, and β
    """
    
    vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
    return vol_estimator


def get_historical_price(ticker: str) -> pd.DataFrame:
    """Get stock's historical price

    Args:
        ticker (str): stock ticker name

    Returns:
        pd.DataFrame: data frame that has historical prices
    """
    start = BDay(1).rollback(date.today() - relativedelta(years=+2))
    res = web.get_data_yahoo(ticker, start, date.today())
    return res["Adj Close"]


def get_volatility(asset_prices: pd.DataFrame) -> VolatilityTracker:
    """Get volatility forecast

    Args:
        asset_prices (pd.DataFrame): stock's historical prices

    Returns:
        VolatilityTracker: _description_
    """
    vol_estimator = _get_garch(asset_prices)
    vol_tracker = volatility_trackers.GARCHVolatilityTracker(
        vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta, asset_prices
    )
    return vol_tracker


def get_yield_curve() -> YieldCurve:
    """Constructing the riskless yield curve based on the current fed funds rate and treasury yields
    
    Returns:
        YieldCurve: yield curve object
    """
    today = date.today()

    # 
    data = web.get_data_fred(
        [
            "DFF",
            "DGS1MO",
            "DGS3MO",
            "DGS6MO",
            "DGS1",
            "DGS2",
            "DGS3",
            "DGS5",
            "DGS7",
            "DGS10",
            "DGS20",
            "DGS30",
        ],
        today - BDay(3),
        today,
    )
    data.dropna(inplace=True)

    cur_date_curve = data.index[-1].date()

    # Convert to percentage points
    data /= 100.0

    # Some adjustments are required to bring FED Funds rate to the same day count convention and compounding frequency
    # as treasury yields (actual/actual with semi-annual compounding):
    # 1. https://www.federalreserve.gov/releases/h15/default.htm -> day count convention for Fed Funds Rate needs
    # to be changed to actual/actual
    # 2. Conversion to APY: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions
    data.DFF *= (
        366 if curves.YieldCurve.is_leap_year(cur_date_curve.year) else 365
    ) / 360  # to x/actual
    data.DFF = 2 * (np.sqrt(data.DFF + 1) - 1)

    offsets = [
        relativedelta(),
        relativedelta(months=+1),
        relativedelta(months=+3),
        relativedelta(months=+6),
        relativedelta(years=+1),
        relativedelta(years=+2),
        relativedelta(years=+3),
        relativedelta(years=+5),
        relativedelta(years=+7),
        relativedelta(years=+10),
        relativedelta(years=+20),
        relativedelta(years=+30),
    ]

    # Define the riskless yield curve
    curve = curves.YieldCurve(
        today,
        offsets,
        data[cur_date_curve : cur_date_curve + BDay()].to_numpy()[0, :],
        compounding_freq=2,
    )
    return curve

def get_dividends(ticker:str)->pd.Series:
    """Get dividends of a stock

    Args:
        ticker (str): stock ticker

    Returns:
        pd.Series: pandas series that contains dates and its dividend amount
    """
    
    ticker_object = yf.Ticker(ticker)
    last_divs = ticker_object.dividends[-1:]

    # An approximate rule for Apple's ex-dividend dates -- ex-dividend date is on the first Friday
    # of the last month of a season if that Friday is the 5th day of the month or later, otherwise
    # it falls on the second Friday of that month.
    idx = (pd.date_range(last_divs.index[0].date(), freq='WOM-1FRI', periods=30)[::3])
    idx = idx.map(lambda dt: dt if dt.day >= 5 else dt+BDay(5))
    divs = pd.Series([last_divs[0]] * len(idx), index=idx, name=ticker + ' Dividends')
    return divs

def price_option(
    volatility: VolatilityTracker,
    stock_price: float,
    strike_price: int,
    risk_free_interest_rate: YieldCurve,
    maturity_date: datetime,
    opt_type: OptionType,
    ticker:str,
) -> OptionsPricer:
    """Function to price an american or european option

    Args:
        volatility (VolatilityTracker):stock's volatility
        stock_price (float): price of a stock
        strike_price (int): strike price
        risk_free_interest_rate (YieldCurve): risk free interest rate
        maturity_date (datetime): maturity date
        opt_type (OptionType): option type(american or european)
        ticker (str): stock ticker

    Returns:
        OptionsPricer: option object that has annual volatility, price, delta, gamma, vega
    """
    holidays = mcal.get_calendar("NYSE").holidays().holidays
    divs = get_dividends(ticker)
    pricer = options.BlackScholesMertonPricer(
        maturity_date,
        volatility,
        strike_price,
        risk_free_interest_rate,
        stock_price,
        ticker=ticker,
        dividends=divs,
        opt_type=opt_type,
        holidays=holidays,
    )
    return pricer

def price_atm_american_option_multi_process(
    ticker:str,
    risk_free_interest_rate: YieldCurve,
    maturity_date: datetime,
    holidays:tuple
) -> OptionsPricer:
    """Function to price an at the money  option

    Args:
        risk_free_interest_rate (YieldCurve): risk free interest rate
        maturity_date (datetime): maturity date
        opt_type (OptionType): option type(american or european)
        ticker (str): stock ticker

    Returns:
        OptionsPricer: option object that has annual volatility, price, delta, gamma, vega
    """
    stock_historical_prices = get_historical_price(ticker)
    stock_price = stock_historical_prices[-1]
    volatility = get_volatility(stock_historical_prices)
    strike_price = int(stock_price) # set strike price equal to stock price 
                                    # so we are pricing an atm option(definition)
    divs = get_dividends(ticker)
    pricer = options.BlackScholesMertonPricer(
        maturity_date,
        volatility,
        strike_price,
        risk_free_interest_rate,
        stock_price,
        ticker=ticker,
        dividends=divs,
        opt_type=OptionType.AMERICAN,
        holidays=holidays,
    )
    return pricer

if __name__ == "__main__":
    start_time = time.perf_counter()
    ticker = "AAPL"
    
    
    strike_price = 225
    

    maturity_date = date(2025, 1, 17)
    
    option_type = OptionType.AMERICAN
    
    pricing_result = price_option(
        strike_price,
        maturity_date,
        option_type,
        ticker
    )
    
    
    end_time = time.perf_counter()  
    total_time = end_time - start_time 
    print(f'Option pricing took {total_time:.4f} seconds') 
    print(pricing_result)
