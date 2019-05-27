

from data_provider import hist_stock_data
from platforms import sandbox_platform
from strategies import dummy_strategy


class market_simulator():
    """Simulate market
    """

    # Init, load & save status
    def __init__(self, data_provider_object, strategy_object, platform_object):
        self.dpo = data_provider_object
        self.so = strategy_object
        self.po = platform_object

    def adjClose_simulation(self):
        df = self.dpo.adjClose()
        symbols = list(df)
        for index, row in df.iterrows():
            for symbol in symbols:
                self.so.set_symbol_value(symbol, row[symbol], row[symbol])
                self.po.set_symbol_value(symbol, row[symbol], row[symbol])  

            actions = self.so.get_recommended_actions()
            self.po.process_actions(actions)

            print(self.po.get_positions())


def unit_test():
    test_data = hist_stock_data(['AAPL'], intersect = True)
    test_data.restrict_date_range('2018-12-01', '2019-01-01')

    test_platform = sandbox_platform(filename = '')
    test_platform.deposit(100000)
    
    test_strategy = dummy_strategy(test_platform)

    sim = market_simulator(test_data, test_strategy, test_platform)
    sim.adjClose_simulation()
    
    print('Balance after simulation: ', test_platform.get_balance_tot())
    
if __name__ == "__main__":
    unit_test()


