
import scipy.optimize as spo
import matplotlib.pyplot as plt
import numpy as np

import data_provider as dp


class sr_portfolio_optimizer():
    """Create a porfolio of provided symbols and date range.
       Use 'optimize_allocation' method to get symbols allocation for best sharp_ratio
    """

    def __init__(self, symbols, from_date = '', to_date = '', provider = 'tiingo', offline = True):
        self.stock_data = dp.hist_stock_data(symbols, intersect = True, provider = provider, offline = offline) # get data
        self.stock_data.restrict_date_range(from_date = from_date, to_date = to_date) # restrict to provided date range
        self.dr = self.stock_data.daily_returns()   # compute daily returns

    def __compute_portfolio_sharp_ratio(self, allocation, args):
        """PRIVATE - Used by optimizer
        Compute and return the sharp ratio of portoflio calculated on daily returns with specified allocation."""
        adr = allocation * self.dr   # alloced_daily_returns
        pdr = adr.sum(axis=1)   # portofolio daily returns
        sr = pdr.mean(axis=0) / pdr.std(axis=0) # sharp_ratio
        if args['inverted']:
            sr = 1 - sr
        return sr

    def provider(self):
        """Give access to provider class"""
        return self.stock_data

    def equally_allocation(self):
        """Return a tuple of equally allocated stocks"""
        return tuple([1/self.dr.shape[1] for x in range(self.dr.shape[1])])

    def sharp_ratio(self, allocation = []):
        """Compute and return the sharp ratio of portoflio calculated on daily returns with specified allocation."""
        if allocation == []:
            allocation = self.equally_allocation()
        return self.__compute_portfolio_sharp_ratio(allocation, {'inverted': False})

    def optimize_allocation(self, disp = False, method='SLSQP'):
        """Run optimizer and return allocation for best sharp_ratio"""
        #Optimizer
        guess = tuple([1/self.dr.shape[1] for x in range(self.dr.shape[1])])
        bnds = tuple([(0, 1.0) for x in range(self.dr.shape[1])])    # define limits foreach allocation
        cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})   #define constraints, sum of allocations must be 1
        best = spo.minimize(self.__compute_portfolio_sharp_ratio, guess, args={'inverted': True},
                                method=method, bounds=bnds, constraints=cons, options={'disp': disp})
        return tuple(best.x)

    def print_portfolio_sharp_ratios(self, best=()):
        """Iterate in allocation printing sharp_ratio for debug purpose
           2/3 stocks portfolio support only
        """
        if self.dr.shape[1] == 2:
            a,c = [],[]
            for x in np.arange(0.0, 1.01, 0.01):
                sharp_ratio = self.__compute_portfolio_sharp_ratio(tuple((x, 1.0-x)), {'inverted': False})
                #print("Allocation:", tuple((x, 1-x)), "SR: ", sharp_ratio)
                a.append(x)
                c.append(sharp_ratio)
            plt.plot(a, c)
            if best != ():
                plt.axvline(x=best[0], color='red')
            plt.show()
        elif self.dr.shape[1] == 3:
            a,b,c = [],[],[]
            for x in np.arange(0.0, 1.01, 0.01):
                for y in np.arange(0.0, 1.01-x, 0.01):
                    if (x+y) < 1:
                        sharp_ratio = self.__compute_portfolio_sharp_ratio(tuple((x, y, 1.0-x-y)), {'inverted': False})
                        a.append(x)
                        b.append(y)
                        c.append(sharp_ratio)
            plt.scatter(a,b, c=c)
            plt.colorbar()
            if best != ():
                plt.axvline(x=best[0], color='red')
                plt.axhline(y=best[1], color='red')
            plt.show()


def optimizer_test_run():
    # Read data

    symbols = ['AAPL', 'GOOG', 'HRTX']

    #tickers =  dp.tickers()
    #symbols = tickers.get_random_tickers('2018-05-01', '2018-05-31', quantity = 2)

    test = sr_portfolio_optimizer(symbols, '2018-05-01', '2018-05-31')

    #plot_data(df)
    print(symbols)
    print("### data head")
    print(test.provider().adjClose().head())
    print()

    # Compute sharp ratio on Guess
    guess = test.equally_allocation() # allocation is equally divided and returned
    sharp_ratio = test.sharp_ratio(guess)
    print("Equally Allocation:", ",".join(format(f, '.3f') for f in guess), "SR (guess): {:.2f}".format(sharp_ratio))
    print()

    #Optimizer
    best = test.optimize_allocation()
    sharp_ratio = test.sharp_ratio(best)
    print("SLSQP  Allocation: ", ",".join(format(f, '.3f') for f in best), "SR (best): {:.2f}".format(sharp_ratio))
    print()

    #Debug
    test.print_portfolio_sharp_ratios(best=best)


if __name__ == "__main__":
    optimizer_test_run()
