

from data_provider import hist_stock_data
from platforms import sandbox_platform
from strategies import dummy_strategy, rolling_mean_strategy


class market_simulator():
    """Simulate market
    """

    # Init, load & save status
    def __init__(self, data_provider_object, strategy_object, platform_object):
        self.dpo = data_provider_object
        self.so = strategy_object
        self.po = platform_object

    def add_sim_step(self, stocks):
        self.sim_steps.append({
            'stocks':stocks,
            'positions': self.po.get_positions(),
            'balance_avail': self.po.get_balance_avail(),
            'balance_total': self.po.get_balance_tot()
        })

    def adjClose_simulation(self):
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

        self.sim_stats['initial_balance'] = self.po.get_balance_tot()
        self.sim_stats['initial_balance_avail'] = self.po.get_balance_avail()
        self.sim_stats['initial_positions'] = self.po.get_positions().copy()
        
        df = self.dpo.adjClose()
        symbols = list(df)
        for index, row in df.iterrows():
            for symbol in symbols:
                self.so.set_symbol_value(symbol, row[symbol], row[symbol])
                self.po.set_symbol_value(symbol, row[symbol], row[symbol])  

            actions = self.so.get_recommended_actions()
            confirms = self.po.process_actions(actions)

            for i in range(len(actions)):
                if confirms[i]['result']:
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
                else:
                    self.sim_stats['failed_actions'] += 1            

            self.add_sim_step(row)

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

        

def unit_test():
    test_data = hist_stock_data(['AAPL'], intersect = True)
    test_data.restrict_date_range('2018-01-01', '2019-01-01')

    test_platform = sandbox_platform(filename = '')
    test_platform.deposit(1000.0)

    #test_strategy = dummy_strategy(test_platform)
    test_strategy = rolling_mean_strategy(test_platform, 21)

    sim = market_simulator(test_data, test_strategy, test_platform)
    sim.adjClose_simulation()
    
    print(sim.sim_stats)

    show_plot(sim)

def show_plot(sim):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    s = sim.get_sim_balance_total()
    s2 = sim.get_sim_stocks('AAPL')

    s = (np.asarray(s)/s[0] - 1) * 100.0
    s2 = (np.asarray(s2)/s2[0] - 1) * 100.0

    max_limit = max([max(s),max(s2)])
    min_limit = min([min(s),min(s2)])
    yticks = np.arange(min_limit,max_limit+5,5, dtype=int)

    t = np.arange(len(s), dtype=int)
    fig, ax = plt.subplots()
    plt.xticks()

    color = 'tab:red'
    ax.set_xlabel('steps')
    ax.set_ylabel('Return (%)', color=color)
    ax.set_ylim([min_limit,max_limit])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks,color=color)
    ax.plot(t, s, color=color)

    color = 'tab:blue'
    ax2 = ax.twinx() 
    ax2.set_ylim([min_limit,max_limit])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks,color=color)
    ax2.set_ylabel('AAPL (%)', color=color) 
    ax2.plot(t, s2, color=color)

    fig.tight_layout()
    plt.show()    

if __name__ == "__main__":
    unit_test()


