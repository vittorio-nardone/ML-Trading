
import platforms

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
            if len(positions) < 2:
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
                   'quantity': 3,
                })
        return actions

def unit_test():
    pass

if __name__ == "__main__":
    unit_test()

