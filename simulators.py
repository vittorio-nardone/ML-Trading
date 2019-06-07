


from data_provider import hist_stock_data, tickers
from platforms import sandbox_platform
from strategies import dummy_strategy, rolling_mean_strategy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class adjClose_market_simulator():
    """Simulate market
    """

    # Init, load & save status
    def __init__(self, data_provider_object, strategy_object, platform_object):
        self.dpo = data_provider_object
        self.so = strategy_object
        self.po = platform_object

    def add_sim_step(self, stocks, actions, confirms):
        self.sim_steps.append({
            'stocks':stocks,
            'positions': self.po.get_positions(),
            'balance_avail': self.po.get_balance_avail(),
            'balance_total': self.po.get_balance_tot(),
            'actions': actions,
            'confirms': confirms
        })

    def simulate(self):
        self.sim_steps = []
        self.sim_stats = {}
        self.sim_stats['opened_positions'] = 0
        self.sim_stats['closed_positions'] = 0
        self.sim_stats['failed_actions'] = 0
        self.sim_stats['max_profit'] = 0
        self.sim_stats['max_loss'] = 0
        self.sim_stats['max_profit_perc'] = 0
        self.sim_stats['max_loss_perc'] = 0
        self.sim_stats['closed_positions_in_loss'] = 0
        self.sim_stats['closed_positions_in_profit'] = 0
        self.sim_stats['max_duration'] = 0

        self.sim_stats['initial_balance'] = self.po.get_balance_tot()
        self.sim_stats['initial_balance_avail'] = self.po.get_balance_avail()
        self.sim_stats['initial_positions'] = self.po.get_positions().copy()

        self.sim_stats['min_balance'] = self.sim_stats['initial_balance']
        self.sim_stats['max_balance'] = self.sim_stats['initial_balance']
        self.sim_stats['min_balance_avail'] = self.sim_stats['initial_balance_avail']
        self.sim_stats['max_balance_avail'] = self.sim_stats['initial_balance_avail']
        
        tick = 0
        df = self.dpo.adjClose()
        symbols = list(df)
        for index, row in df.iterrows():
            tick += 1
            for symbol in symbols:
                self.so.set_symbol_value(symbol, row[symbol], row[symbol], tick)
                self.po.set_symbol_value(symbol, row[symbol], row[symbol])  

            actions = self.so.get_recommended_actions()
            confirms = self.po.process_actions(actions, time_ref=tick)

            for i in range(len(actions)):
                if confirms[i]['result']:
                    actions[i]['confirmed'] = True
                    if actions[i]['is_open']:
                        self.sim_stats['opened_positions'] += 1
                    else:
                        self.sim_stats['closed_positions'] += 1 
                        if confirms[i]['profit_loss']>0:
                            self.sim_stats['closed_positions_in_profit'] += 1
                            self.sim_stats['max_profit'] = max([self.sim_stats['max_profit'] , confirms[i]['profit_loss']])
                            self.sim_stats['max_profit_perc'] = max([self.sim_stats['max_profit_perc'] , confirms[i]['profit_loss_perc']])
                        else:
                            self.sim_stats['closed_positions_in_loss'] += 1
                            self.sim_stats['max_loss'] = min([self.sim_stats['max_loss'] , confirms[i]['profit_loss']])
                            self.sim_stats['max_loss_perc'] = min([self.sim_stats['max_loss_perc'] , confirms[i]['profit_loss_perc']])
                        self.sim_stats['max_duration'] = max([self.sim_stats['max_duration'], confirms[i]['avg_duration']])
                else:
                    actions[i]['confirmed'] = False
                    self.sim_stats['failed_actions'] += 1            

            self.add_sim_step(row, actions, confirms)

            self.sim_stats['min_balance'] = min([self.sim_stats['min_balance'], self.po.get_balance_tot()])
            self.sim_stats['max_balance'] = max([self.sim_stats['max_balance'], self.po.get_balance_tot()])
            self.sim_stats['min_balance_avail'] = min([self.sim_stats['min_balance_avail'], self.po.get_balance_avail()])
            self.sim_stats['max_balance_avail'] = max([self.sim_stats['max_balance_avail'], self.po.get_balance_avail()])

        self.sim_stats['final_balance'] = self.po.get_balance_tot()
        self.sim_stats['final_balance_avail'] = self.po.get_balance_avail()
        self.sim_stats['final_positions'] = self.po.get_positions().copy()


    # Simulation related
    def get_sim_steps(self, column):
        return list(map(lambda x : x[column], self.sim_steps))

    def get_sim_balance_total(self):
        return self.get_sim_steps('balance_total')

    def get_sim_balance_avail(self):
        return self.get_sim_steps('balance_avail')

    def get_sim_positions(self):
        return self.get_sim_steps('positions')

    def get_sim_stocks(self, symbol):
        stocks = self.get_sim_steps('stocks')
        return list(map(lambda x : x[symbol], stocks))

    def get_actions(self):
        return self.get_sim_steps('actions')


def unit_test():
    #symbols = ['AAPL']
    symbols = ['AAPL','AMZN','GOOG','IBM','SPY']

    date_from, date_to = '2018-01-01', '2019-06-01'

    #tickers_list = tickers()
    #symbols = tickers_list.get_random_tickers(date_from, date_to, quantity = 30)
    #symbols = ['PTGX','PPC','USAT','AGNC','LNDC','BPFH','OBAS','DENN','GSBC',
    #           'CNSL','BGCP','AEIS','IIJIY','LMB','EMCF','HIFS','CASI','GNMK','UPL']
    print(symbols)

    window_size = 21
    profit_limit = 0.25
    loss_limit = -0.10
    use_short = True
    max_stock_allocation = 0.10

    test_data = hist_stock_data(symbols, intersect = True, offline = True)
    test_data.restrict_date_range(date_from, date_to)

    test_platform = sandbox_platform(filename = '')
    test_platform.deposit(2000.0)

    #test_strategy = dummy_strategy(test_platform)
    test_strategy = rolling_mean_strategy(test_platform, window_size, max_stock_allocation = max_stock_allocation, profit_limit = profit_limit, loss_limit = loss_limit, use_short=use_short)

    sim = adjClose_market_simulator(test_data, test_strategy, test_platform)
    sim.simulate()
    
    print(sim.sim_stats)

    plot_simulation(sim, symbols, window_size)

def plot_simulation(sim, symbols, window_size, show_mom = False):
    s1 = sim.dpo.adjClose()
    rm1 = sim.dpo.rolling_mean(window_size)
    rstd1 = sim.dpo.rolling_std(window_size)

    for symbol in symbols:
        #s2 = np.asarray(sim.get_sim_stocks(symbol))
        s2 = np.asarray(s1[symbol])
        rm = np.asarray(rm1[symbol].values.tolist())
        rstd = np.asarray(rstd1[symbol].values.tolist())

        mom_func = lambda x: (x[-1] - x[0])
        mom = s1[symbol].rolling(10).apply(mom_func, raw=True)  
        mom = np.asarray(mom.values.tolist())

        upper_band = rm + 2 * rstd
        lower_band = rm - 2 * rstd

        max_limit = max(s2)
        min_limit = min(s2)

        t = np.arange(len(s2), dtype=int)
        fig, ax = plt.subplots()
        plt.xticks()

        color = 'tab:red'
        ax.set_ylim([min_limit,max_limit])
        ax.set_title(symbol)
        ax.plot(t, s2, color=color)

        color = 'tab:blue'
        ax.plot(t, rm, color=color)
        ax.plot(t, upper_band, color=color)
        ax.plot(t, lower_band, color=color)

        actions = sim.get_actions()
        p = 0
        for a in actions:
            if len(a) > 0:
                for i in a:
                    if i['symbol'] == symbol:
                        if i['confirmed'] == True:
                            if i['is_open']:
                                marker = 'b'
                                if i['is_long'] :
                                    marker += '^'
                                else:
                                    marker += 'v'                    
                            else:
                                if i['reason'] == 'stop_loss':
                                    marker = 'rs'
                                elif i['reason'] == 'take_profit':
                                    marker = 'ys'
                                else:
                                    marker = 'gs'
                        else:
                            marker = 'k*'
                        plt.plot(p, s2[p], marker)
            p += 1

        if show_mom:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:cyan'
            ax2.plot(t, mom, color=color)
    

    fig.tight_layout()
    plt.show()  

if __name__ == "__main__":
    unit_test()


