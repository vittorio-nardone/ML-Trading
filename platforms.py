

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

        self.status['balance_avail'] = 0.0
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
    def history_add(self, operation, symbol, quantity, open_value, on_screen = True, time_ref = 0):
        row = [str(datetime.datetime.utcnow()), str(self.status['balance_avail']), str(len(self.status['positions'])), str(self.opened_positions_value()), operation, str(symbol), str(quantity), str(open_value), str(time_ref)]
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

    def opened_positions_value(self, symbols = []):
        value = 0.0
        for p in self.status['positions']:
            if (symbols == []) or (p['symbol'] in symbols):
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

    def add_position(self, symbol, quantity, is_long = True, time_ref = 0):
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
                'open_time': time_ref
            })
            self.status['balance_avail'] -= (value * quantity)
            if is_long:
                self.history_add('OPEN_LONG',symbol,quantity,value, time_ref = time_ref)
            else:
                self.history_add('OPEN_SHORT',symbol,quantity,value, time_ref = time_ref)
            return {'result':True}
        else:
            return {'result':False}
    
    def close_position(self, symbol, quantity, time_ref = 0):
        new_positions = []
        new_quantity = quantity
        profit_loss = 0.0
        opened_value = 0.0
        avg_duration = 0
        affected = 0
        to_close = []
        for position in self.status['positions']:
            if position['symbol'] == symbol:
                if new_quantity <= position['quantity']:
                    if position['is_long']:
                        value = self.get_sell_symbol_value(position['symbol'])
                        self.status['balance_avail'] += (value * new_quantity)
                        profit_loss += (value - position['open_value']) * new_quantity
                        opened_value += position['open_value'] * new_quantity
                    else:
                        value = self.get_buy_symbol_value(position['symbol'])
                        self.status['balance_avail'] +=  ((position['open_value'] - value) * new_quantity) + (position['open_value'] * new_quantity) 
                        profit_loss += (position['open_value'] - value) * new_quantity
                        opened_value += value * new_quantity
                    self.history_add('CLOSE',position['symbol'],new_quantity,value,time_ref=time_ref)
                    position['quantity'] -= new_quantity
                    new_quantity = 0
                    if position['quantity']>0:
                        new_positions.append(position)    
                else:
                    to_close.append(position)
                    new_quantity -= position['quantity']
                avg_duration += time_ref - position['open_time'] 
                affected += 1                      
            else:
                new_positions.append(position)

        if new_quantity == 0:
            self.status['positions'] = new_positions
            for position in to_close:
                    if position['is_long']:
                        value = self.get_sell_symbol_value(position['symbol'])
                        self.status['balance_avail'] += (value * position['quantity'])
                        profit_loss += (value - position['open_value']) * position['quantity']
                        opened_value += position['open_value'] * position['quantity']
                    else:
                        value = self.get_buy_symbol_value(position['symbol'])
                        self.status['balance_avail'] +=  ((position['open_value'] - value) * position['quantity']) + (position['open_value'] * position['quantity']) 
                        profit_loss += (position['open_value'] - value) * position['quantity']
                        opened_value += value * position['quantity']
                    self.history_add('CLOSE',position['symbol'],position['quantity'],value,time_ref=time_ref)
            return {'result':True, 'profit_loss':profit_loss, 'profit_loss_perc': profit_loss/opened_value, 'affected_position': affected, 'avg_duration': avg_duration/affected}
        else:
            return {'result':False}

    def process_actions(self, actions, time_ref = 0):
        confirms = []
        for action in actions:
            if action['is_open']:
                confirms.append(self.add_position(action['symbol'], action['quantity'], action['is_long'], time_ref = time_ref))
            else:
                confirms.append(self.close_position(action['symbol'], action['quantity'], time_ref = time_ref))
        return confirms


def unit_test():
    test = sandbox_platform(filename = '')
    test.deposit(500)
    test.set_symbol_value('EUR_USD',100,99)
    test.add_position('EUR_USD', 1, is_long = False)
    test.set_symbol_value('EUR_USD',80,79)
    test.close_position('EUR_USD', 1)
    test.save_status()

if __name__ == "__main__":
    unit_test()