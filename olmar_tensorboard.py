import sys
import logbook
import numpy as np
import pandas as pd

from zipline.finance import commission

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()

# DOW 30
STOCKS = ['300640']

#from tensorboard import TensorBoard


# On-Line Portfolio Moving Average Reversion
# More info can be found in the corresponding paper:
# http://icml.cc/2012/papers/168.pdf
def initialize(algo, eps=1, window_length=5):
    algo.stocks = STOCKS
    algo.sids = [algo.symbol(symbol) for symbol in algo.stocks]
    algo.m = len(algo.stocks)
    algo.price = {}
    algo.b_t = np.ones(algo.m) / algo.m
    algo.last_desired_port = np.ones(algo.m) / algo.m
    algo.eps = eps
    algo.init = True
    algo.days = 0
    algo.window_length = window_length

    algo.set_commission(commission.PerShare(cost=0))

    try:
        Nparams = len(algo.algo_params)
        if Nparams == 2:
            UseParams = True
        else:
            print 'len context.algo_params is', Nparams, ' expecting 2'
    except Exception as e:
        print 'context.params not passed', e
    if UseParams:
        print 'Setting Algo parameters via passed algo_params'
        algo.eps = algo.algo_params['eps']
        algo.tb_log_dir= algo.algo_params['logdir']
    '''
    if algo.tb_log_dir:
        algo.tensorboard = TensorBoard(log_dir=algo.tb_log_dir)
    else:
        algo.tensorboard = None
    '''

def handle_data(algo, data):
    algo.days += 1
    if algo.days < algo.window_length:
        return

    if algo.init:
        rebalance_portfolio(algo, data, algo.b_t)
        algo.init = False
        return

    m = algo.m

    x_tilde = np.zeros(m)

    # find relative moving average price for each asset
    mavgs = data.history(algo.sids, 'price', algo.window_length, '1d').mean()
    for i, sid in enumerate(algo.sids):
        price = data.current(sid, "price")
        # Relative mean deviation
        x_tilde[i] = mavgs[sid] / price

    ###########################
    # Inside of OLMAR (algo 2)
    x_bar = x_tilde.mean()

    # market relative deviation
    mark_rel_dev = x_tilde - x_bar

    # Expected return with current portfolio
    exp_return = np.dot(algo.b_t, x_tilde)
    weight = algo.eps - exp_return
    variability = (np.linalg.norm(mark_rel_dev)) ** 2

    # test for divide-by-zero case
    if variability == 0.0:
        step_size = 0
    else:
        step_size = max(0, weight / variability)

    b = algo.b_t + step_size * mark_rel_dev
    b_norm = simplex_projection(b)
    np.testing.assert_almost_equal(b_norm.sum(), 1)

    rebalance_portfolio(algo, data, b_norm)

    # update portfolio
    algo.b_t = b_norm

    # record something to show that these get logged
    # to tensorboard as well:
    algo.record(x_bar=x_bar)
    '''
    if algo.tensorboard is not None:
        # record algo stats to tensorboard
        algo.tensorboard.log_algo(algo)
    '''

def rebalance_portfolio(algo, data, desired_port):
    # rebalance portfolio
    desired_amount = np.zeros_like(desired_port)
    current_amount = np.zeros_like(desired_port)
    prices = np.zeros_like(desired_port)

    if algo.init:
        positions_value = algo.portfolio.starting_cash
    else:
        positions_value = algo.portfolio.positions_value + \
                          algo.portfolio.cash

    for i, sid in enumerate(algo.sids):
        current_amount[i] = algo.portfolio.positions[sid].amount
        prices[i] = data.current(sid, "price")

    desired_amount = np.round(desired_port * positions_value / prices)

    algo.last_desired_port = desired_port
    diff_amount = desired_amount - current_amount

    for i, sid in enumerate(algo.sids):
        algo.order(sid, diff_amount[i])

def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain
       Implemented according to the paper: Efficient projections onto the
       l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
       Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
       Optimization Problem: min_{w}\| w - v \|_{2}^{2}
       s.t. sum_{i=1}^{m}=z, w_{i}\geq 0
       Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
       Output: Projection vector w
       :Example:
       >>> proj = simplex_projection([.4 ,.3, -.4, .5])
       >>> proj  # doctest: +NORMALIZE_WHITESPACE
       array([ 0.33333333, 0.23333333, 0. , 0.43333333])
       >>> print(proj.sum())
       1.0
       Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
       Python-port: Copyright 2013 by Thomas Wiecki (thomas.wiecki@gmail.com).
       """

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p + 1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho + 1)])
    w = (v - theta)
    w[w < 0] = 0
    return w

# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(context=None, results=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    results.portfolio_value.plot(ax=ax)
    ax.set_ylabel('Portfolio value (USD)')
    plt.show()


# Note: this if-block should be removed if running
# this algorithm on quantopian.com

from zipline.utils.run_algo import _run
from pandas.tslib import Timestamp
import os

def set_args(eps,logdir):
    parsed = {}
    parsed['initialize'] = initialize
    parsed['handle_data'] = handle_data
    parsed['before_trading_start'] = None
    parsed['analyze'] = None
    parsed['algotext'] = None
    parsed['defines'] = ()
    parsed['capital_base'] = 1000000
    parsed['data'] = None
    parsed['bundle'] = 'my-yahoo-equities-bundle'
    # parsed['bundle']='YAHOO'
    # parsed['bundle_timestamp']=None
    parsed['bundle_timestamp'] = pd.Timestamp.utcnow()
    parsed['start'] = Timestamp('2017-05-01 13:30:00+0000', tz='UTC')
    parsed['end'] = Timestamp('2017-05-31 13:30:00+0000', tz='UTC')
    #parsed['algofile'] = open('D:\\workspace\\algotrading\\MyAlgo\\optim\\spearmint-try.py')
    parsed['algofile'] = None
    parsed['data_frequency'] = 'daily'
    parsed['bm_symbol'] = None
    parsed['print_algo'] = False
    parsed['output'] = 'os.devnull'
    parsed['local_namespace'] = None
    parsed['environ'] = os.environ

    # Below what we expect spearmint to pass us
    # parsed['algo_params']=[47,88.7,7.7]
    # D={}
    # D['timeperiod']=10
    # D['nbdevup']=1.00
    # D['nbdevdn']=1.00
    parsed['algo_params'] = {'eps':eps,'logdir':logdir}
    return  parsed

if __name__ == "__main__":
    for eps in [1.0, 1.25, 1.5]:
        args = set_args(eps,'/tmp/olmar/Dow-30/eps = %.2f' % eps)
        perf = _run(**args)
        #olmar.tb_log_dir = '/tmp/olmar/Dow-30/eps = %.2f' % eps
        print '-' * 100
        print '/tmp/olmar/Dow-30/eps = %.2f' % eps
        results = _run(args)

    '''
    # Set the simulation start and end dates.
    start = datetime(2004, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)

    # Load price data from yahoo.
    data = load_from_yahoo(stocks=STOCKS, indexes={}, start=start, end=end)
    data = data.dropna()

    for eps in [1.0, 1.25, 1.5]:
        # Create and run the algorithm.
        olmar = TradingAlgorithm(handle_data=handle_data,
                                 initialize=initialize,
                                 identifiers=STOCKS)
        olmar.eps = eps
        olmar.tb_log_dir = '/tmp/olmar/Dow-30/eps = %.2f' % eps
        
        print '-'*100
        print olmar.tb_log_dir

        results = olmar.run(data)

        # Plot the portfolio data.
        #analyze(results=results)
    '''