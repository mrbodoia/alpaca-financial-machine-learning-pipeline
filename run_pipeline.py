import requests
import pandas_market_calendars as mcal
import time
import pickle
import math
import pandas as pd

def get_date_bars(ticker_symbol, open_timestamp, close_timestamp):

    # set the url for pulling bar data
    base_url = 'https://data.alpaca.markets/v1/bars/minute'

    # set the request headers using our api key/secret
    request_headers = {'APCA-API-KEY-ID': '<YOUR_KEY_HERE>', 'APCA-API-SECRET-KEY': '<YOUR_SECRET_HERE>'}

    # set the request params for the next request
    request_params = {'symbols': ticker_symbol, 'limit': 1000, 'start': open_timestamp.isoformat(), 'end': close_timestamp.isoformat()}

    # get the response
    date_bars = requests.get(base_url, params=request_params, headers=request_headers).json()[ticker_symbol]

    # if the date on the response matches the closing date for the day, throw the candle away (since it technically happens after the close)
    if date_bars[-1]['t'] == int(close_timestamp.timestamp()):
        date_bars = date_bars[:-1]

    # return the bars for the date
    return date_bars


def get_all_bars(ticker_symbol):

    # get a list of market opens and closes for each trading day from 2015 onwards
    trading_days = mcal.get_calendar('NYSE').schedule(start_date='2015-01-01', end_date='2020-08-31')

    # initialize an empty list of all bars
    all_bars = []

    # for each day in our list of trading days...
    for i in range(len(trading_days)):

        # get the time at the start of the request
        request_time = time.time()

        # get the list of bars for the next day
        next_bars = get_date_bars(ticker_symbol, trading_days['market_open'][i], trading_days['market_close'][i])
    
        # print a log statement
        print(f'Got bars for {next_bars[-1]["t"]}')
    
        # add the next bars to our growing list of all bars
        all_bars += next_bars
    
        # sleep to ensure that no more than 200 requests occur per 60 seconds
        time.sleep(max(request_time + 60/200 - time.time(), 0))

    # return the list of all bars
    return all_bars


def get_dollar_bars(time_bars, dollar_threshold):

    # initialize an empty list of dollar bars
    dollar_bars = []

    # initialize the running dollar volume at zero
    running_volume = 0

    # initialize the running high and low with placeholder values
    running_high, running_low = 0, math.inf

    # for each time bar...
    for i in range(len(time_bars)):

        # get the timestamp, open, high, low, close, and volume of the next bar
        next_timestamp, next_open, next_high, next_low, next_close, next_volume = [time_bars[i][k] for k in ['t', 'o', 'h', 'l', 'c', 'v']]

        # get the midpoint price of the next bar (the average of the open and the close)
        midpoint_price = (next_open + next_close)/2

        # get the approximate dollar volume of the bar using the volume and the midpoint price
        dollar_volume = next_volume * midpoint_price

        # update the running high and low
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)

        # if the next bar's dollar volume would take us over the threshold...
        if dollar_volume + running_volume >= dollar_threshold:

            # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
            bar_timestamp = next_timestamp + 60
            
            # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
            dollar_bars += [{'timestamp': bar_timestamp, 'high': running_high, 'low': running_low, 'close': next_close}]

            # reset the running volume to zero
            running_volume = 0

            # reset the running high and low to placeholder values
            running_high, running_low = 0, math.inf

        # otherwise, increment the running volume
        else:
            running_volume += dollar_volume

    # return the list of dollar bars
    return dollar_bars


def add_feature_columns(bars_df, period_length):

    # get the price vs ewma feature
    bars_df[f'feature_PvEWMA_{period_length}'] = bars_df['close']/bars_df['close'].ewm(span=period_length).mean() - 1

    # get the price vs cumulative high/low range feature
    bars_df[f'feature_PvCHLR_{period_length}'] = (bars_df['close'] - bars_df['low'].rolling(period_length).min()) / (bars_df['high'].rolling(period_length).max() - bars_df['low'].rolling(period_length).min())

    # get the return vs rolling high/low range feature
    bars_df[f'feature_RvRHLR_{period_length}'] = bars_df['close'].pct_change(period_length)/((bars_df['high']/bars_df['low'] - 1).rolling(period_length).mean())

    # get the convexity/concavity feature
    bars_df[f'feature_CON_{period_length}'] = (bars_df['close'] + bars_df['close'].shift(period_length))/(2 * bars_df['close'].rolling(period_length+1).mean()) - 1

    # get the rolling autocorrelation feature
    bars_df[f'feature_RACORR_{period_length}'] = bars_df['close'].rolling(period_length).apply(lambda x: x.autocorr()).fillna(0)

    # return the bars df with the new feature columns added
    return bars_df


def get_feature_matrix(dollar_bars):

    # convert the list of bar dicts into a pandas dataframe
    bars_df = pd.DataFrame(dollar_bars)

    # number of bars to aggregate over for each period
    period_lengths = [4, 8, 16, 32, 64, 128, 256]

    # for each period length
    for period_length in period_lengths:

        # add the feature columns to the bars df
        bars_df = add_feature_columns(bars_df, period_length)

    # prune the nan rows at the beginning of the dataframe
    bars_df = bars_df[period_lengths[-1]:]

    # filter out the high/low/close columns and return 
    return bars_df[[column for column in bars_df.columns if column not in ['high', 'low', 'close']]]


# download and save the raw data to a pickle file (only need to do this once)
#pickle.dump(get_all_bars('SPY'), open('SPY.pkl', 'wb'))

# load the raw data from the pickle file
time_bars = pickle.load(open('SPY.pkl', 'rb'))

# convert the time bars to dollar bars
dollar_bars = get_dollar_bars(time_bars, 50000000)

# construct the feature matrix from the dollar bars
feature_matrix = get_feature_matrix(dollar_bars)
