# """ main file that pulls the latest information from polygon python client"""
# Initial imports
import os
import sys
import pandas as pd
from dotenv import load_dotenv
# Load environment variables
#import polygon as settings 
from datetime import date, datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from polygon import RESTClient

# FORCES THE PROGRAM TO FIND THE CORRECT .ENV FILE
dotenv_path = os.path.join(os.path.dirname(__file__),'.env')
load_dotenv(dotenv_path)
# print(f"API KEY: {os.getenv('polygon_api_key')}")

from local_settings import *

try:
    sys.path.append(source_code_path)
    print(source_code_path)
    polygon_api_key = os.getenv("polygon_api_key")
    print(polygon_api_key)
except:
    print("SourceCode absolute path not found!")
    sys.path.append(input("Enter SourceCode folder absolute path"))

        
class MyRESTClient(RESTClient):

    def __init__(self, auth_key: str=os.getenv('polygon_api_key'), timeout:int=5):
        
        # super().__init__(os.getenv("polygon_api_key"))
        super().__init__(auth_key)
        self.markets = ['crypto', 'stocks', 'fx']
        retry_strategy = Retry(total=10,
                               backoff_factor=10,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount('https://', adapter)


    def get_tickers(self, market:str=None) -> pd.DataFrame:
        if not market in self.markets:
            raise Exception(f'Market must be one of {self.markets}.')

        resp = self.reference_tickers_v3(market=market)
        if hasattr(resp, 'results'):
            df = pd.DataFrame(resp.results)

            while hasattr(resp, 'next_url'):
                resp = self.reference_tickers_v3(next_url=resp.next_url)
                df = df.append(pd.DataFrame(resp.results))

            if market == 'crypto':
                # Only use USD pairings.</em>
                df = df[df['currency_symbol'] == 'USD']
                df['name'] = df['base_currency_name']
                df = df[['ticker', 'name', 'market', 'active']]

            df = df.drop_duplicates(subset='ticker')
            return df
        return None

    def get_bars(self, market:str=None, ticker:str=None, multiplier:int=1,
                 timespan:str='minute', from_:date=None, to:date=None) -> pd.DataFrame:
        """
        market: 'crypto', 'stocks', 'fx'
        
        ticker: use get_tickers to get the list of all the tickers

        timespan: minute

        from_ / to: date format YYYY-mm-dd (i.e. 2022-09-10)
        
        returns: Open High Low Close Volume dataframe (by minutes)
        """

        if not market in self.markets:
            raise Exception(f'Market must be one of {self.markets}.')

        if ticker is None:
            raise Exception('Ticker must not be None.')

        from_ = from_ if from_ else date(2000,1,1)
        to = to if to else date.today()

        if market == 'crypto':
            resp = self.crypto_aggregates(ticker, multiplier, timespan,
                                          from_.strftime('%Y-%m-%d'), to.strftime('%Y-%m-%d'),
                                          limit=50000)
            df = pd.DataFrame(resp.results)
            last_minute = 0
            while resp.results[-1]['t'] > last_minute:
                last_minute = resp.results[-1]['t'] 
                # Last minute in response
                last_minute_date = datetime.fromtimestamp(last_minute/1000).strftime('%Y-%m-%d')
                resp = self.crypto_aggregates(ticker, multiplier, timespan,
                                          last_minute_date, to.strftime('%Y-%m-%d'),
                                          limit=50000)
                new_bars = pd.DataFrame(resp.results)
                df = df.append(new_bars[new_bars['t'] > last_minute])

            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o':'open',
                                    'h':'high',
                                    'l':'low',
                                    'c':'close',
                                    'v':'volume',
                                    'vw':'vwap',
                                    'n':'transactions'})
            df = df[['date','open','high','low','close','volume']]

            return df
        return None