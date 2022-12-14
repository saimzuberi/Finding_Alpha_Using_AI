# Finding_Alpha

Finding Alpha is a journey into finding a trading strategy that would allow for getting Alpha on a crypto currency. The system is designed in a manner that it takes Polygon API which is free source of data and allow us to fetch data of a wide verity of equities. 

# Installation Guide

Following are the imports necesssary for the note book to run. 

                ! pip instal polygon-api-client          
                ! pip install pandas
                ! pip install numpy
                ! pip install hvplot
                ! pip install matplotlib 
                ! pip intsall talib
                ! pip install spicy 
                ! pip install requests
                ! pip install urllib3
                ! pip install sklearn
                ! pip install tensorflow
                ! pip install keras

# API KEY 

API key can be obtained from https://polygon.io/docs/stocks/getting-started
by creating a new account. 

## Polygon API

Polygon API comes with pre canned functions and what we have implemented is a process to download data which bypasses the restrction on the number of requrest that we can pull on minute basis. 
    
        def __init__(self, auth_key: str=settings['api_key'], timeout:int=5):
        super().__init__(auth_key)
        retry_strategy = Retry(total=10,
                               backoff_factor=10,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount('https://', adapter)

Further to the downloading strategy, the API has been customised to work with Stocks, FX and Crypto Curriencies. 

        def get_tickers(self, market:str=None) -> pd.DataFrame:
        if not market in markets:
            raise Exception(f'Market must be one of {markets}.')

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

Finally, we have defined the function that would allow us to download a customizable data set that starting from 1 minute onwards for all Equities. 

    def get_bars(self, market:str=None, ticker:str=None, multiplier:int=1,
                 timespan:str='minute', from_:date=None, to:date=None) -> pd.DataFrame:

        if not market in markets:
            raise Exception(f'Market must be one of {markets}.')

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
                last_minute = resp.results[-1]['t'] # Last minute in response</em>
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

The result of the same is the dataset that would be a dataframe with all the information required. 

        start = datetime(2000,1,1)
        client = MyRESTClient(settings['api_key'])
        df = client.get_bars(market='crypto', ticker='X:BTCUSD', from_=start)
        df

The resulting data frame looks like as below. 

![Alt text](../Documentation/images/Dataframe%20BTC.png "BTC DATAFRAME")

Implementing and HVPLOT for the same results. 

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/BTC_HVPLOT.png "BTC DATAFRAME")

## CURRENCIES SELECTION

Following are the rules which have been used for selecting the currencies for the project. 

a. Market Cap
b. Should not have a direct relationship with USD in 1:1 ratio i.e. (USDC and BUSD)

As a result following is the list of currencies which have been selected. 

            Bitcoin
            Etherium
            Bianance
            Polygon
            Ripple
            Cardano

## Trading Indicidators

### Close percentage
        BTC_df['ClosePct']=(BTC_df['close']-BTC_df['low'])/(BTC_df['high']-BTC_df['low'])

### Rolling slope over 30 bars
        BTC_df['rolling_slope'] = BTC_df['close'].rolling(window=minutes_back).apply(get_slope, raw=False)
Where Minute back is defined as 30 bars

        def get_slope(array):
            y = np.array(array)
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return slope

### Range Z - Score
        BTC_df['RangeZscore']=(BTC_df['Range']-BTC_df['RangeMa'])/df['RangeSD']

### SMA Fast over 30 bars over close
        BTC_df['SMA_Fast'] = BTC_df['close'].rolling(window=short_window).mean()/BTC_df['close']
Where short window is defined as 30 minutes.

### SMA Slow over 100 bars over close
        BTC_df['SMA_Slow'] = BTC_df['close'].rolling(window=long_window).mean()/BTC_df['close']
Where short window is defined as 100 minutes.

### RSI
        BTC_df['RSI']=talib.RSI(BTC_df['close'], timeperiod=30)

### Volume percentage
        BTC_df['Volume_PCT'] = BTC_df['volume'].pct_change()

### Bollinger Band High over close
        BTC_df['Bollinger High'] = (BTC_df['rolling_mean'] + (BTC_df['rolling_std'] * 2))/BTC_df['close']

### Bollinger Bank Low over close
        BTC_df['Bollinger Low'] = (BTC_df['rolling_mean'] - (BTC_df['rolling_std'] * 2))/BTC_df['close']

Note : All of the above have rationalized to close 

As a result the indicator dataframe is as below for all cryptocurrencies

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Indicators_Dataframe.png "BTC DATAFRAME")

# Algorithimic Trading

We have defined the following aglorithm for trading. 

        signals_df['Signal'] = 0.0
        RSI_Buy = 70
        RSI_Sell = 30
        Slope_Buy = 0.1
        Slope_sell= -0.1


        for index, row in signals_df.iterrows():
            if row["ClosePct"] > row["Bollinger Low"] and row["SMA_Fast"] > row["SMA_Slow"] and row["RSI"] >= RSI_Buy and row['rolling_slope'] > row["SMA_Fast"] :
            signals_df.loc[index, "Signal"] = -1.0
            if  row["ClosePct"] < row["Bollinger High"]and row["SMA_Fast"] < row["SMA_Slow"] and RSI_Sell >= row["RSI"] and row['rolling_slope'] < row["SMA_Fast"]:
            signals_df.loc[index,"Signal"] = 1.0

Which takes into account the following
#### Sell Signal
a. Where close percentage is greater than Bollinger Low 
b. SMA Fast is greater than SMA Slow
c. RSI is greater than 70
d. Rolling Slope is greater than SMA Fast

#### Buy Signal 
a. Where close percentage is greater than Bollinger High
b. SMA Slow is greater than SMA Fast
c. RSI is less than or equal to 30
d. Rolling Slope is less than SMA Fast

The following are the results for BTC and ETH

## BTC ALGORITHM TRADES

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Trading_Algo_BTC.png "BTC DATAFRAME")

### BTC Portfolio Value

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/BTC_Portfolio_Value.png "BTC DATAFRAME")

### BTC Risk / Reward Strategy


![Alt text](../Finding_Alpha_Using_AI/Documentation/images/BTC_Risk_Reward.png "BTC DATAFRAME")

### BTC Trades

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/BTC_Trades.png "BTC DATAFRAME")

### BTC Profit & Loss

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/BTC_ProfitLoss.png "BTC DATAFRAME")

## ETH ALGORITHM TRADES


![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Trading_Algo_ETH.png "BTC DATAFRAME")

### ETH Portfolio Value

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/ETH_Portfolio_Value.png "BTC DATAFRAME")

### ETH Risk / Reward Strategy


![Alt text](../Finding_Alpha_Using_AI/Documentation/images/ETH_Risk_Reward.png "BTC DATAFRAME")

### ETH Trades

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/ETH_Trades.png "BTC DATAFRAME")

### ETH Profit & Loss

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/ETH_ProfitLoss.png "BTC DATAFRAME")

# USING AI TO PREDICT SIGNALS

## Logistic Regresssion Results for Buy and Sell Signals

### Logistic Regression Accuracy Score

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Logisstic_Regression_Accuracy%20Score.png "BTC DATAFRAME")

### Logisstic Regression Confusion Matrix

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Logistic_Regression%20Confussion%20Matrix.png "BTC DATAFRAME")

### Classification Report

                       precision  recall  f1-score   support      
        -1.0            1.00      1.00      1.00      4981       
        1.0             0.00       0.00     0.00        19    accuracy                              1.00     5000   
        macro avg       0.50      0.50      0.50      5000
        weighted avg    0.99      1.00      0.99      5000


# NEURAL NETWORK 

Setting up the Neural Network Model
![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Screenshot%202022-12-14%20at%205.19.58%20PM.png "BTC DATAFRAME")

![Alt text](../Finding_Alpha_Using_AI/Documentation/images/Screenshot%202022-12-14%20at%205.21.40%20PM.png "BTC DATAFRAME")

## Neural Network Result

        157/157 - 0s - 
        loss: 822395.6250 - 
        mse: 822395.6250 - 311ms/epoch - 2ms/step
        Loss: 822395.625, 
        Accuracy: 822395.625

