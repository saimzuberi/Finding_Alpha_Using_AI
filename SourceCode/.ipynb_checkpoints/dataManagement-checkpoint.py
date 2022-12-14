from marketData import MyRESTClient
import os
from local_settings import *
from datetime import datetime
import pandas as pd
import sys
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

class DatabaseManagement(MyRESTClient):
    def __init__(self,market='crypto') -> None:
        MyRESTClient.__init__(self)
        if market not in self.markets:
            sys.exit("Wrong Market >> Shutting Down Program (Restart Required)")
        self.market = market
        # CHECK IF THE FODLERS EXIST FOR THE MARKET
        self.src_db = os.path.join(data_storage_path,self.market)
        # IF Subfolders do not exist: Create them
        subfolders = ['Database','OHLC','Stats']
        if not os.path.exists(self.src_db):
            [os.makedirs(os.path.join(self.src_db,f)) for f in subfolders]
        self.src_database = os.path.join(self.src_db,subfolders[0])
        self.src_ohlc = os.path.join(self.src_db,subfolders[1])
        self.src_stats = os.path.join(self.src_db,subfolders[2])
        self.today = datetime.today().strftime("%Y-%m-%d")
    
    @property
    def ticker_universe(self,update=False):
        """
        Get the entire ticker universe based on the selected/instanciated market
        """
        src_ticker_df = os.path.join(self.src_database,'tickers.csv')
        if not os.path.exists(src_ticker_df) or update:
            # GET THE LATEST TICKERS UNIVERSE IN THE MARKET
            ticker_df = self.get_tickers(self.market)
            if self.market=='crypto':
                # Removing X: from the front of the ticker name
                #ticker_df.ticker = ticker_df.ticker.apply(lambda x: x.lstrip("X:"))
            # STORING THE DataFrame into Google Drive
            ticker_df.to_csv(src_ticker_df)
        else:
            ticker_df = pd.read_csv(src_ticker_df,index_col=0)
        return ticker_df

    def ohlc_update(self, ticker:str,multiplier:int=1,timespan:str='day',start:datetime=None,end:datetime=None,update=False):
        """
        if market=crypto >> no need to put 'X:' at the start of the ticker i.e. instead of ticker='X:BTCUSD' do ticker='BTCUSD'
        ticker: Crypto Currencies, stocks, forex based on the market selected
        multiplier: integer >> default to 1
        timespan: minute / day >> default to day
        start: datetime object >> Default None; will go as early as 2000 01 01
        end: datetime object >> Default None; automatically set to today's date
        update: Boolean True/False >> Default False | if True will enforce pulling in fresh data and Merge it with archived data
        """
        ######## TEMPORARY BLOCK DOWNLOAD FOR STOCKS AND FOREX ###########
        if self.market in ['stocks','fx']:
            print("Functionality for stocks and forex coming soon...")
            return
        ##################################################################

        timeframe = f"{multiplier}{timespan[0]}"


        if timespan not in ['day','minute']:
            print("ERROR: Wrong timespan >> Must be 'day' or 'minute'")
            return 

        # CHECK against the entire universe for the tickers
        ticker_df = self.ticker_universe

        # Check if the ticker exists in the universe
        if not ticker in ticker_df.ticker.tolist():
            print(f"ERROR!: Ticker {ticker} not found in the universe Market: {self.market}")
            return

        # Input Perp for and Update
        if self.market=='crypto':
            _ticker = "X:"+ticker
        
        timeframe_folder_path = os.path.join(self.src_ohlc,timeframe)
        # CHECK: TIMEFRAME FOLDER EXISTS
        if not os.path.exists(timeframe_folder_path):
            os.mkdir(timeframe_folder_path)

        ticker_path = os.path.join(timeframe_folder_path,f"{ticker}.csv")
        # If no file exists in google drive >> download and store the file
        if not os.path.exists(ticker_path) or update:
            df = self.get_bars(self.market,_ticker,multiplier,timespan,from_=start,to=end)
            df.to_csv(ticker_path)
            # return OHLC dataFrame
            return df
        else:
            return pd.read_csv(ticker_path,index_col=0)
        
    