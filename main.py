from pathlib import Path
import pandas as pd
from typing import List
from multiprocessing import Pool, cpu_count
from option_pricing_api import get_yield_curve,price_atm_american_option_multi_process
from pricing.options import BlackScholesMertonPricer
import pandas_market_calendars as mcal
from datetime import date

def load_stock_list() -> List[str]:
    """Read stock list from csv file"""
    project_dir = Path(__file__).parent
    stock_file_path = project_dir.joinpath("data", "stock_list.csv")
    stock_list = pd.read_csv(stock_file_path)
    return stock_list["ticker"].to_list()

def format_pricing_result_as_dataframe(pricing_results:List[BlackScholesMertonPricer])->pd.DataFrame:
    final_result = []
    for pricing_result in pricing_results:
        
        res = {}
        res["Price"] = pricing_result.price
        res["Delta"] = pricing_result.delta
        res["Gamma"] = pricing_result.gamma
        res["Vega"] = pricing_result.vega
        res["Rho"] = pricing_result.rho 
        res["Theta"] = pricing_result.theta
        res["Volatility"] = pricing_result.annual_volatility
        final_result.append(res)
    return pd.DataFrame(final_result)
    
def main():
    stock_list = load_stock_list()
    holidays = mcal.get_calendar("NYSE").holidays().holidays
    risk_free_interest_rate = get_yield_curve()
    maturity_date = date(2025,1,17)
    with Pool(processes=2) as pool:
        pricing_results = pool.starmap(price_atm_american_option_multi_process,[(x, risk_free_interest_rate,maturity_date,holidays) for x in stock_list])

    results_df = format_pricing_result_as_dataframe(pricing_results)
    results_df['ticker'] = stock_list
    results_df.to_csv("pricing_result.csv")
   


if __name__ == "__main__":
    main()