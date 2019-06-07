
import platforms
import pandas as pd
import numpy as np

import random

class dummy_strategy():
    """
    """

    # Init, load & save status
    def __init__(self, platform_object):
        self.po = platform_object
        self.symbols = {}

    def set_symbol_value(self, symbol, buy_price, sell_price):
        self.symbols[symbol] = [buy_price, sell_price]

    def get_recommended_actions(self):
        actions = []
        positions = self.po.get_positions()
        stock = list(self.symbols.keys())
        if len(stock) > 0:
            if (random.randint(0, 10) > 8):
                if len(positions) == 0:
                    actions.append({
                       'is_open': True, 
                       'symbol': stock[0],
                       'quantity': 2,
                       'is_long': True,
                    })
                else:
                    actions.append({
                       'is_open': False, 
                       'symbol': stock[0],
                       'quantity': 2,
                    })

        return actions

class rolling_mean_strategy():
    """
    """
    # Init, load & save status
    def __init__(self, platform_object, window, max_stock_allocation = 0.10, profit_limit = 0.02, loss_limit = -0.05, use_short = True, mom_window = 10, mom_short_th = -0.15):
        self.po = platform_object
        self.stock_values = pd.DataFrame()
        self.window = window
        self.profit_limit = profit_limit
        self.loss_limit = loss_limit
        self.use_short = use_short
        self.mom_window = mom_window
        self.mom_short_th = mom_short_th 
        self.max_stock_allocation = max_stock_allocation

        self.buy_long_at_cross = {}
        self.buy_short_at_cross = {}

    def set_symbol_value(self, symbol, buy_price, sell_price, time_ref):
        if (symbol not in list(self.stock_values)):
            self.stock_values[symbol] = np.nan
            self.buy_long_at_cross[symbol] = False
            self.buy_short_at_cross[symbol] = False
        if (time_ref in self.stock_values.index): 
            self.stock_values.loc[time_ref][symbol] = buy_price
        else:
            df = pd.DataFrame([[buy_price]], columns=[symbol], index=[time_ref])
            self.stock_values = self.stock_values.append(df)

    def get_recommended_actions(self):
        actions = []

        if self.stock_values.shape[0] > self.window:
            symbols = list(self.stock_values)
            positions = self.po.get_positions()

            total_balance = self.po.get_balance_tot()
            avail_balance = self.po.get_balance_avail()

            for symbol in symbols:
                current_value = self.stock_values[symbol].iloc[-1]
                rma = self.stock_values[symbol].rolling(self.window).mean()
                rstd = self.stock_values[symbol].rolling(self.window).std()
                bb_high = rma + 2 * rstd
                bb_low = rma - 2 * rstd
               
                current_bb_high = bb_high.iloc[-1]
                current_bb_low = bb_low.iloc[-1]

                mom = (current_value - self.stock_values[symbol].iloc[-self.mom_window]) / current_value

                symbol_alloc = self.po.opened_positions_value(symbols = [symbol]) / total_balance
                
                if symbol_alloc < self.max_stock_allocation:
                    max_alloc_value = (self.max_stock_allocation - symbol_alloc) * total_balance
                    quantity = int(min([max_alloc_value,avail_balance]) / current_value)
                    if quantity > 0:
                        if (current_value >= current_bb_low) and (self.buy_long_at_cross[symbol]) and (mom > 0):
                            self.buy_long_at_cross[symbol] = False
                            self.buy_short_at_cross[symbol] = False
                            actions.append({
                            'is_open': True, 
                            'symbol': symbol,
                            'quantity': quantity,
                            'is_long': True,
                            })  
                        elif (current_value >= current_bb_low) and (self.buy_short_at_cross[symbol]) and (mom < self.mom_short_th):
                            self.buy_long_at_cross[symbol] = False
                            self.buy_short_at_cross[symbol] = False
                            actions.append({
                            'is_open': True, 
                            'symbol': symbol,
                            'quantity': quantity,
                            'is_long': False,
                            })    
                        elif (current_value < current_bb_low):
                            self.buy_long_at_cross[symbol] = True
                            if self.use_short:
                                self.buy_short_at_cross[symbol] = True

                for p in positions:
                    if p['symbol'] == symbol:
                        if p['is_long']:
                            profit_loss = current_value / p['open_value'] - 1
                            close_at_flag = (current_value >= current_bb_high) 
                        else:
                            profit_loss = p['open_value'] / current_value - 1
                            close_at_flag = (current_value <= current_bb_low)

                        stop_loss_flag = (profit_loss <= self.loss_limit)
                        take_profit_flag = (profit_loss > self.profit_limit)

                        if close_at_flag:
                            close_reason = 'price'
                        elif stop_loss_flag:
                            close_reason = 'stop_loss'
                        elif take_profit_flag:
                            close_reason = 'take_profit'
                        else:
                            close_reason = ''

                        if (close_reason != ''):
                                actions.append({
                                'is_open': False, 
                                'symbol': symbol,
                                'quantity': p['quantity'],
                                'reason': close_reason
                                })

        return actions




def unit_test():
    pass

if __name__ == "__main__":
    unit_test()

