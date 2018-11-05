import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like  #bugfix, pandas_datareader fix to be compatible with pandas 0.23
import pandas_datareader.data as pdr
import datetime
import os.path
import matplotlib.pyplot as plt
import numpy as np

## Add new supported providers here
class tiingo_stock_data():
    tiingo_api_token = '2fd8a287a3e17f5b9df11354cc45d93f93a7b6df'

    def __init__(self, symbol):
        self.symbol = symbol

    def download(self, save = True):
        """Download and save data from provider"""
        df = pdr.get_data_tiingo(self.symbol, api_key=self.tiingo_api_token)
        if save:
            df.to_csv("raw_data/tiingo/{}.csv".format(self.symbol))
        return df

    def load(self):
        """Load file from storage"""
        file_path = "raw_data/tiingo/{}.csv".format(self.symbol)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col = ['symbol','date'], parse_dates = True)
        else:
            df = self.download()
        return df

    def supported_tickers(self):
        file_path = "raw_data/tiingo/supported_tickers.csv"
        return pd.read_csv(file_path, parse_dates = True)

class tickers():
    """Helper for tickers pick up"""
    def __init__(self):
        pass

    def get_random_tickers(self, from_date, to_date, quantity = 1, provider = 'tiingo', assetType = 'Stock', exchange = 'NASDAQ'):
        """Get random tickers from provider, according to assetType and Exchange"""
        if provider == 'tiingo':
            dp = tiingo_stock_data('')
        else:
            raise ValueError('Unsupported provider "{}"'.format(provider))
        ##

        st = dp.supported_tickers()
        st = st.loc[st['exchange'] == exchange]
        st = st.loc[st['assetType'] == assetType]
        st = st.loc[st['startDate'] <= from_date]
        st = st.loc[st['endDate'] >= to_date]

        idx = np.random.randint(st.shape[0], size = quantity)

        return st.iloc[idx]['ticker'].values


class hist_stock_data():
    def __init__(self, symbols = [], provider = 'tiingo', offline = True, intersect = False):
        """Init class and optionally add stock data"""
        if symbols != []:
            self.add_symbols(symbols, provider = provider, offline = offline, intersect = intersect)

    ## Symbols management section
    def add_symbols(self, symbols, provider = 'tiingo', offline = True, intersect = False):
        """Add stock data to df dataframe.
            symbols -> array of stock identifier
            provider -> data provider
            offline -> if False, download fresh data from provider (and update local files)
        """
        for symbol in symbols:
            ## Add new supported providers here
            if provider == 'tiingo':
                dp = tiingo_stock_data(symbol)
            else:
                raise ValueError('Unsupported provider "{}"'.format(provider))
            ##

            if offline:
                df_temp = dp.load()
            else:
                df_temp = dp.download()

            if hasattr(self, 'df'):
                self.df = pd.concat([self.df, df_temp])
            else:
                self.df = df_temp

        if intersect:
            self.remove_outside_date_intersection()

    def remove_symbols(self, symbols):
        """Remove stock data from df dataframe"""
        self.df = self.df.drop(symbols, level='symbol')
        self.df.index = self.df.index.remove_unused_levels()

    def update_symbols(self, symbols, provider = 'tiingo'):
        """Remove old stock data from df dataframe and download fresh data"""
        self.remove_symbols(symbols)
        self.add_symbols(symbols, provider = provider, offline = False)
    # end of symbols management section

    # Dates section
    def date_range(self, symbols = []):
        """Return absolute min/max timestamps of stock data"""
        return min(list(self.df.index.levels[1])), max(list(self.df.index.levels[1]))

    def date_range_intersection(self, symbols = []):
        """Get min/max timestamps of stock data intersection
           symbols -> if empty, check all symbols
        """
        max, min = pd.Timestamp.now(), pd.Timestamp(1970, 1, 1)
        if symbols == []:
            symbols = list(self.df.index.levels[0])

        # Compute min and max values
        for symbol in symbols:
            tmp_min, tmp_max = self.df.loc[symbol].index.min(), self.df.loc[symbol].index.max()
            if tmp_min > min:
                min = tmp_min
            if tmp_max < max:
                max = tmp_max

        # Check if found data range is available for all stocks
        if min > max:
            raise ValueError('No intersection for symbols "{}"'.format(symbols))

        return min, max

    def remove_date_range(self, from_date, to_date):
        """Remove all stock data in date range"""
        range = pd.date_range(from_date, to_date)
        self.df = self.df.drop(range, level='date')
        self.df.index = self.df.index.remove_unused_levels()

    def remove_outside_date_intersection(self):
        """Remove all stock data outside intersection of all symbols"""
        min,max = self.date_range_intersection()
        self.remove_date_range(pd.Timestamp(1970, 1, 1), min - pd.Timedelta(days=1))
        self.remove_date_range(max + pd.Timedelta(days=1), pd.Timestamp.now())

    def restrict_date_range(self, from_date = '', to_date = ''):
        """Remove all stock data outside provided dates"""
        if from_date != '':
            self.remove_date_range(pd.Timestamp(1970, 1, 1), pd.Timestamp(from_date) - pd.Timedelta(days=1))
        if to_date != '':
            self.remove_date_range(pd.Timestamp(to_date) + pd.Timedelta(days=1), pd.Timestamp.now())

    # end of dates section

    # Stats section
    def daily_returns(self, symbols = [], relative = False):
        """Compute daily returns of provided symbols
           symbols -> if empty, work on all symbols
        """
        adjClose = self.adjClose(symbols = symbols, relative = relative)

        dr = (adjClose / adjClose.shift(1)) - 1
        dr.iloc[0,:] = 0  #azzero prima riga (shift introduce NaN)
        return dr

    def comulative_returns(self, symbols = [], relative = False):
        """Compute comulative returns of provided symbols
           symbols -> if empty, work on all symbols
        """
        adjClose = self.adjClose(symbols = symbols, relative = relative)

        cr = adjClose.iloc[-1,:] / adjClose.iloc[0,:] - 1
        return cr

    def sharp_ratio(self, symbols = []):
        """Compute and return the sharp ratio calculated on daily returns
        """
        dr = self.daily_returns(symbols = symbols)
        sr = dr.mean(axis=0) / dr.std(axis=0)
        return sr

    def rolling_mean(self, window, symbols = [], relative = False):
        """Return rolling mean of given values, using specified window size."""
        adjClose = self.adjClose(symbols = symbols, relative = relative)
        return adjClose.rolling(window).mean()

    def rolling_std(self, window, symbols = [], relative = False):
        """Return rolling std of given values, using specified window size."""
        adjClose = self.adjClose(symbols = symbols, relative = relative)
        return adjClose.rolling(window).std()

    def bollinger_bands(self, window, std_size = 2, symbols = [], relative = False):
        """Return upper and lower Bollinger Bands."""
        rm = self.rolling_mean(window, symbols = symbols, relative = relative)
        rstd = self.rolling_std(window, symbols = symbols, relative = relative)
        upper_band = rm + std_size * rstd
        lower_band = rm - std_size * rstd
        return upper_band, lower_band


    ### Following indicators are stock price independent (ML)

    def bb(self, window, std_size = 2, symbols = []):
        """Return Bollinger Bands indicator (-1 -> 1) of given values, using specified window size."""
        adjClose = self.adjClose(symbols = symbols)
        bb_func = lambda x: (x[-1] - x.mean()) / (std_size * x.std())
        return adjClose.rolling(window).apply(bb_func, raw=True)

    def sma(self, window, symbols = []):
        """Return Simple moving avarage of given values, using specified window size."""
        adjClose = self.adjClose(symbols = symbols)
        sma_func = lambda x: (x[-1] / x.mean()) -1
        return adjClose.rolling(window).apply(sma_func, raw=True)

    def momentum(self, window, symbols = []):
        """Return momentum, using specified window size."""
        adjClose = self.adjClose(symbols = symbols)
        mom_func = lambda x: (x[-1] / x[0]) -1
        return adjClose.rolling(window).apply(mom_func, raw=True)

    def indicators_grid(self, symbol, rolling_windows = [10], bb_sizes = [2]):
        """Return indicators of specified symbol."""

        sma_func = lambda x: (x[-1] / x.mean()) -1
        mom_func = lambda x: (x[-1] / x[0]) -1

        tmp = self.df.loc[symbol].copy()

        for w in rolling_windows:
            for std_size in bb_sizes:
                bb_func = lambda x: (x[-1] - x.mean()) / (std_size * x.std())
                tmp['bb{}_{}'.format(std_size, w)] = tmp['adjClose'].rolling(w).apply(bb_func, raw=True)
            tmp['sma_{}'.format(w)] = tmp['adjClose'].rolling(w).apply(sma_func, raw=True)
            tmp['mom_{}'.format(w)] = tmp['adjClose'].rolling(w).apply(mom_func, raw=True)

        tmp.dropna(inplace = True)

        tmp['dr'] = tmp['adjClose'] / tmp['adjClose'].shift(1) -1
        tmp.fillna(0, inplace = True)

        return tmp

    # end of stats section

    # Easy access to data
    def __get_data_column(self, column, symbols = [], relative = False):
        """Return column of specified symbols."""
        if symbols == []:
            symbols = list(self.df.index.levels[0])

        for symbol in symbols:
            tmp = self.df.loc[symbol][column].to_frame(name = symbol)
            if relative:
                tmp = tmp / tmp.iloc[0]

            if 'col' in locals():
                col = col.join(tmp, how='outer')
            else:
                col = tmp
        return col

    def adjClose(self, symbols = [], relative = False):
        """Return adjClose of specified symbols."""
        return self.__get_data_column('adjClose',symbols = symbols, relative = relative)

    def adjVolume(self, symbols = [], relative = False):
        """Return adjVolume of specified symbols."""
        return self.__get_data_column('adjVolume',symbols = symbols, relative = relative)

    # end of Easy access to data


def bb_test_run():
    test = hist_stock_data(['PPG'], intersect = True)
    test.restrict_date_range('2017-01-01', '2018-01-01')

    rm_SPY = test.rolling_mean(20, symbols = ['PPG'], relative = True)
    upper_band, lower_band = test.bollinger_bands(20, symbols = ['PPG'], relative = True)

    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = test.adjClose(['PPG'], relative = True).plot(title="Bollinger Bands", label='')
    rm_SPY.plot(ax=ax)
    upper_band.plot(ax=ax)
    lower_band.plot(ax=ax)

    bb = test.bb(20)
    bb.plot(ax=ax)

    ax.legend(["PPG", "Rolling mean", "Upper band", "Lower band"], loc='upper left');

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def test_run():
    test = hist_stock_data(['AAPL'], intersect = True)
    test.restrict_date_range('2017-01-01', '2018-01-01')

    tmp = test.indicators_grid('AAPL', rolling_windows = [3,10,30,90], bb_sizes = [1.5,2,2.5,3])

    print(temp.head())


if __name__ == "__main__":
    test_run()
