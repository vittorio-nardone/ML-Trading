

import data_provider as dp


class prediction_model():

    def __init__(self, indicators_grid):
        self.ig = indicators_grid

        self.days = 10                 #previsione a 10 giorni, comulative_returns
        self.buy_profit_th = 0.03      #soglia di comulative_returns oltre la quale conviene comprare
        self.short_profit_th = 0.03    #soglia di comulative_returns (* -1) oltre la quale conviene short
        self.stop_loss_warn = 0.02     #soglia intervento stop_loss (percentuale di adjClose)

    def compute_dataset(self):
        self.ds = self.ig.copy()
        # comulative_returns nei successivi 10 giorni
        self.ds['cr']  = (self.ds['adjClose'] / self.ds['adjClose'].shift(self.days) -1).shift(-1 * self.days)
        # valore massimo e minimo (in % di escursione) nei successivi 10 giorni
        self.ds['min'] = self.ds['adjLow'].rolling(self.days).min().shift(-1 * self.days) / self.ds['adjClose'] - 1
        self.ds['max'] = self.ds['adjHigh'].rolling(self.days).max().shift(-1 * self.days) / self.ds['adjClose'] - 1
        # indicatore buy se comulative_returns nei successivi 10 giorni > buy_profit_th
        self.ds['buy'] = self.ds['cr'] > self.buy_profit_th
        # indicatore short se comulative_returns nei successivi 10 giorni < -1 * short_profit_th
        self.ds['short'] = self.ds['cr'] < (-1 * self.short_profit_th)

        # indicatore possibile intervento stop_loss (quando valore stock supera nella direzione sbagliata la soglia stop_loss_warn)
        self.ds['stop'] = (self.ds['buy'] == True) & (self.ds['min'] < -1 * self.stop_loss_warn) | (self.ds['short'] == True) & (self.ds['max'] > self.stop_loss_warn)

        self.ds.dropna(inplace = True)
        # rimuove colonne non utili per il training del modello
        self.ds.drop(['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'close', 'divCash', 'high', 'low', 'open', 'splitFactor', 'volume'], axis=1, inplace = True)

    def get_datasets(self, train_size = 0.8):
        train_count = int(self.ds.shape[0] * train_size)
        return self.ds.iloc[:train_count], self.ds.iloc[train_count:]






def test_run():
    test = dp.hist_stock_data(['AAPL'], intersect = True)
    test.restrict_date_range('2010-01-01', '2018-01-01')

    tmp = test.indicators_grid('AAPL', rolling_windows = [3,10,30,90], bb_sizes = [1.5,2,2.5,3])

    model = prediction_model(tmp)
    model.compute_dataset()

    train_ds, test_ds = model.get_datasets()
    print(train_ds, test_ds)

if __name__ == "__main__":
    test_run()
