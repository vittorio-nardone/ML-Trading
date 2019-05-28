
import platforms
import pandas as pd

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
    def __init__(self, platform_object, window, profit_limit = 0.10, loss_limit = -0.10):
        self.po = platform_object
        self.stock_values = pd.DataFrame()
        self.window = window
        self.profit_limit = profit_limit
        self.loss_limit = loss_limit

    def set_symbol_value(self, symbol, buy_price, sell_price):
        df = pd.DataFrame([[buy_price]], columns=[symbol])
        self.stock_values = self.stock_values.append(df)

    def get_recommended_actions(self):
        actions = []

        if self.stock_values.shape[0] > self.window:
            symbols = list(self.stock_values)
            positions = self.po.get_positions()

            rma = self.stock_values[symbols[0]].rolling(self.window).mean()
            rstd = self.stock_values[symbols[0]].rolling(self.window).std()
            bb_high = rma + 2 * rstd
            bb_low = rma - 2 * rstd
               
            current_rma = rma.iloc[-1:][0]
            current_bb_high = bb_high.iloc[-1:][0]
            current_bb_low = bb_low.iloc[-1:][0]            

            current_value = self.stock_values[symbols[0]].iloc[-1:][0]
            previous_value = self.stock_values[symbols[0]].iloc[-2:-1][0]

            if len(positions) == 0:
                if (current_value <= current_bb_low):
                    self.close_at = current_rma
                    actions.append({
                       'is_open': True, 
                       'symbol': symbols[0],
                       'quantity': 2,
                       'is_long': True,
                    })  
                elif (current_value >= current_bb_high):
                    self.close_at = current_rma
                    actions.append({
                       'is_open': True, 
                       'symbol': symbols[0],
                       'quantity': 2,
                       'is_long': False,
                    })  

            else:
                for p in positions:
                    profit_loss = current_value / p['open_value'] - 1
                    #if (profit_loss >= self.profit_limit) or (profit_loss <= self.loss_limit):
                    if p['is_long']:
                        if (current_value >= self.close_at):
                            actions.append({
                                'is_open': False, 
                                'symbol': symbols[0],
                                'quantity': p['quantity']
                            })
                    else:
                        if (current_value <= self.close_at):
                            actions.append({
                                'is_open': False, 
                                'symbol': symbols[0],
                                'quantity': p['quantity']
                            })                        

        return actions




def unit_test():
    pass

if __name__ == "__main__":
    unit_test()

