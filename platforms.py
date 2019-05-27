

import json
import os 
import datetime

import csv   

from random import randint


class sandbox_platform():
    """Simulate a trading platform
    """

    # Init, load & save status
    def __init__(self, filename = 'sandbox_platform.json', activity_log_file = 'platform.log'):
        self.status = {}
        self.symbols = {}

        self.status['balance_avail'] = 0
        self.status['positions'] = []
        
        if activity_log_file != '':
            self.activity_file = open(activity_log_file, 'a')
            self.activity_writer = csv.writer(self.activity_file)

        self.history_add('INIT','','','')

        if filename != '':
            self.load_status(filename) 

    def save_status(self, filename = 'sandbox_platform.json'):
        status_dict = {'status': self.status, 'symbols': self.symbols}
        with open(filename, 'w') as outfile:  
            json.dump(status_dict, outfile)

    def load_status(self, filename):
        if (os.path.isfile(filename)):
            with open(filename) as json_file:  
                status_dict = json.load(json_file)
                self.status = status_dict['status']
                self.symbols = status_dict['symbols']
                self.history_add('LOAD','','','')
                

    # Operation history
    def history_add(self, operation, symbol, quantity, open_value, on_screen = True):
        row = [str(datetime.datetime.utcnow()), str(self.status['balance_avail']), str(len(self.status['positions'])), str(self.opened_positions_value()), operation, symbol, str(quantity), str(open_value)]
        msg = ','.join(row)
        if hasattr(self, 'activity_writer'): 
            self.activity_writer.writerow(row)   
        if on_screen:
            print(msg)

    # Market Prices
    def set_symbol_value(self, symbol, buy_price, sell_price):
        self.symbols[symbol] = [buy_price, sell_price]

    def get_buy_symbol_value(self, symbol):
        if symbol in self.symbols:
            return self.symbols[symbol][0] 
        else:
            return None

    def get_sell_symbol_value(self, symbol):
        if symbol in self.symbols:
            return self.symbols[symbol][1] 
        else:
            return None

    # Get info
    def get_balance_tot(self):
        return self.status['balance_avail'] + self.opened_positions_value()

    def opened_positions_value(self):
        value = 0
        for p in self.status['positions']:
            if p['is_long']:
                value += p['quantity'] * self.get_sell_symbol_value(p['symbol']) 
            else:
                value += ((p['open_value'] - self.get_buy_symbol_value(p['symbol'])) * p['quantity']) + (p['open_value'] * p['quantity']) 
        return value

    def get_balance_avail(self):
        return self.status['balance_avail']

    def get_positions(self):
        return self.status['positions']

    # Actions (return True if ok)
    def deposit(self, quantity):
        self.status['balance_avail'] += quantity
        self.history_add('DEPOSIT','',quantity,'')
        return True

    def withdraw(self, quantity):
        if (quantity <= self.status['balance_avail']):
            self.status['balance_avail'] -= quantity
            self.history_add('WITHDRAW','',quantity,'')
            return True
        else:
            return False

    def add_position(self, symbol, quantity, is_long = True):
        if is_long:
            value = self.get_buy_symbol_value(symbol) 
        else:
            value = self.get_sell_symbol_value(symbol) 

        if (self.status['balance_avail']) >= (value * quantity):
            self.status['positions'].append({  
                'symbol': symbol,
                'quantity': quantity,
                'is_long': is_long,
                'open_value': value,
                'position_id': len(self.status['positions'])
            })
            self.status['balance_avail'] -= (value * quantity)
            if is_long:
                self.history_add('OPEN_LONG',symbol,quantity,value)
            else:
                self.history_add('OPEN_SHORT',symbol,quantity,value)
            return True
        else:
            return False
    
    def close_position(self, symbol, quantity):
        new_positions = []
        new_quantity = quantity
        to_close = []
        for position in self.status['positions']:
            if position['symbol'] == symbol:
                if new_quantity <= position['quantity']:
                    if position['is_long']:
                        value = self.get_sell_symbol_value(position['symbol'])
                        self.status['balance_avail'] += (value * new_quantity)
                    else:
                        value = self.get_buy_symbol_value(position['symbol'])
                        self.status['balance_avail'] +=  ((position['open_value'] - value) * new_quantity) + (position['open_value'] * new_quantity) 
                    self.history_add('CLOSE',position['symbol'],new_quantity,value)
                    position['quantity'] -= new_quantity
                    new_quantity = 0
                    if position['quantity']>0:
                        new_positions.append(position)    
                else:
                    to_close.append(position)
                    new_quantity -= position['quantity']
            else:
                new_positions.append(position)

        if new_quantity == 0:
            self.status['positions'] = new_positions
            for position in to_close:
                    if position['is_long']:
                        value = self.get_sell_symbol_value(position['symbol'])
                        self.status['balance_avail'] += (value * position['quantity'])
                    else:
                        value = self.get_buy_symbol_value(position['symbol'])
                        self.status['balance_avail'] +=  ((position['open_value'] - value) * position['quantity']) + (position['open_value'] * position['quantity']) 
                    self.history_add('CLOSE',position['symbol'],position['quantity'],value)
            return True
        else:
            return False

    def process_actions(self, actions):
        confirms = []
        for action in actions:
            if action['is_open']:
                confirms.append(self.add_position(action['symbol'], action['quantity'], action['is_long']))
            else:
                confirms.append(self.close_position(action['symbol'], action['quantity']))
        return confirms


def unit_test():
    test = sandbox_platform(filename = '')
    test.deposit(500)
    test.set_symbol_value('EUR_USD',100,99)
    test.add_position('EUR_USD', 1, is_long = False)
    test.set_symbol_value('EUR_USD',80,79)
    test.close_position(0)
    test.save_status()

if __name__ == "__main__":
    unit_test()