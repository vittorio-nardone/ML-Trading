
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
        pass

def unit_test():
    pass

if __name__ == "__main__":
    unit_test()

