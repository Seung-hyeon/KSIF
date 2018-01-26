"""
CSCV module

 CSCV(Combinatorially Symmetric Cross Valdiation)

 백테스트하면서 좋은 전략을 찾으려고 할 때 과거 데이터는 고정되어있기 때문에
 언제나 오버피팅(Overfitting)할 가능성이 있다.
 오버피팅 문제를 피하기 위해서 사용하는 방법이 CSCV이다.
 시계열 데이터를 일정한 등분으로 쪼개 In Sample(IS)과 Out of Sample(OOS)로 나눠
 모든 조합에 대해 IS에서 가장 좋은 전략이 OOS에서 여전히 좋은지 체크한다.
 그 때 IS에서 제일 좋은 전략이 OOS에서 median 이하의 performance를 보여줄 경우를
 모두 카운트하여 오버피팅확률(Probability of Backtest Overfitting)이라 정의한다.
 이를 통해 전략 선택 과정이 얼마나 오버피팅 되었는지 확인할 수 있다.

 (David H. Bailey et al, Journal of Computational Finance, 2015)

"""
import KSIF as kf
from KSIF.core.data import get
from KSIF.core.ffn import calc_port_return
from KSIF.core import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pyprind

__author__ = 'Seung Hyeon Yu'
__email__ = 'rambor12@business.kaist.ac.kr'


class CSCV:
    """
    CSCV class
    """

    def __init__(self, strategy_set, prices, part=16, measure='avg return', perf_func=None):
        # Set price index as a pd.Timestamp
        if not isinstance(prices.index[0], pd.Timestamp):
            prices = prices.reset_index()
            prices = prices.set_index('DATE')

        print('\n')
        self.part = part
        self.strategy_set = strategy_set
        self.PBO = None
        self.stability = []
        self.ranks = []
        self.performance = {'IS': [], 'OOS': []}
        self.strategies = []
        self.combination = 0
        self.log = "IS Combination\tIndex\tIS Best perf\t    OOSperf / OOS bestperf\trank\n"
        self.start = str(np.array(prices.index)[0])[:10]
        self.end = str(np.array(prices.index)[-1])[:10]
        self.measure = measure

        # Setting
        prices = prices
        port_set = []
        eq_weights = []
        bar = pyprind.ProgBar(len(strategy_set), title='Initializing.')
        for strategy in strategy_set:
            # port에 future return 붙어 있음
            port = strategy.build_port(prices)
            # check whether eq weight or not
            eq_weight = len(set(port.groupby(level=0)['weights'].agg(lambda x: len(set(x)) == 1))) == 1
            port_set.append(port)
            eq_weights.append(eq_weight)
            bar.update()

        date = np.array_split(list(prices.index.drop_duplicates()), self.part)
        print('done.\n')

        # Do Validation with all combinations
        combination = list(itertools.combinations(range(self.part), int(self.part / 2)))
        self.combination = len(combination)
        print(self.log)
        for IS in combination:
            OOS = list(set(range(self.part)) - set(IS))
            # select best strategy
            # In Sample
            perf = []
            for port, eq_weight in zip(port_set, eq_weights):
                if perf_func is None:
                    ret = calc_port_return(port,
                                           dates=np.concatenate([date[i] for i in IS], axis=0),
                                           eq_weight=eq_weight)
                    perf.append(calc_perf(ret, measure))
                else:
                    perf.append(perf_func(port))
            self.performance['IS'].append(np.nanmax(perf))
            best_ind = perf.index(np.nanmax(perf))
            IS_ranks = np.array(perf).argsort().argsort()

            # Out Of Sample
            perf = []
            for port, eq_weight in zip(port_set, eq_weights):
                if perf_func is None:
                    ret = calc_port_return(port,
                                           dates=np.concatenate([date[i] for i in OOS], axis=0),
                                           eq_weight=eq_weight)
                    perf.append(calc_perf(ret, measure))
                else:
                    perf.append(perf_func(port))
            self.performance['OOS'].append(perf[best_ind])
            OOS_ranks = np.array(perf).argsort().argsort()

            # get rank
            rank = np.array(perf).argsort().argsort()[best_ind]
            self.ranks.append(rank)
            self.strategies.append(strategy_set[best_ind])
            IS_perf = utils.prettyfloat(self.performance['IS'][-1] * 100) + " %"
            OOS_perf = utils.prettyfloat(self.performance['OOS'][-1] * 100) + " % /" + utils.prettyfloat(
                np.nanmax(perf) * 100) + " % "
            log = str(IS) + "\t" + "{: 5d}".format(best_ind) + "\t" + IS_perf + "\t" + OOS_perf + "\t" + str(rank)
            print(log)
            self.log += log + "\n"
            self.PBO = (np.array(self.ranks) < len(self.strategy_set) / 2).sum() / self.combination
            '''
            Stability란?
             IS rank vector r(IS)와 OOS rank vector r(OOS)가 있을 때 다음과 같은 Norm을 정의한다.

             |r(IS) - r(OOS)|_(stab) := ( (N-1)N(N+1)/6 - Sum(r(IS)_i - r(OOS)_i)^2 ) / ( (N-1)N(N+1)/6 )

             이는 rank가 얼마나 변하는지 알려주는 -1 ~ 1의 척도로써
             1 : 100% Consistent, rank가 변하지 않는다.
             0 : Fully Random, rank가 랜덤하게 움직인다.
             -1: OOS의 rank가 IS의 rank와 정반대로 움직인다.
            '''
            N = len(self.strategy_set)
            SSE = np.vectorize(lambda x: x ** 2)(np.array(IS_ranks) - np.array(OOS_ranks)).sum()
            stab_rand = (N - 1) * N * (N + 1) / 6
            self.stability.append((stab_rand - SSE) / stab_rand)

    def print(self, option='Summary'):
        result = ""
        if option == 'Summary':
            result = "\n------------------ CSCV SUMMARY ------------------"
            result += "\n DATE\t\t\t: " + self.start + " ~ " + self.end
            result += "\n Strategy Name\t\t: " + self.strategy_set[0].name
            result += "\n # of Strategties\t: " + str(len(self.strategy_set))
            result += "\n # of Time Partition\t: " + str(self.part)
            result += "\n # of Combinations\t: " + str(self.combination)
            result += "\n Performance Measure\t: " + str(self.measure)
            result += "\n\n                     [RESULT]                    "
            result += "\n IS  performance AVG\t: " + utils.prettyfloat(np.mean(self.performance['IS']) * 100) + " %"
            result += "\n OOS performance AVG\t: " + utils.prettyfloat(np.mean(self.performance['OOS']) * 100) + " %"
            result += "\n PBO \t\t\t: " + utils.prettyfloat(self.PBO * 100) + " %"
            result += "\n Rank Stability  \t: " + utils.prettyfloat(np.array(self.stability).mean() * 100) + " %"
            result += "\n--------------------------------------------------"
        elif option == 'Histogram':
            pass
        elif option == 'Scatter':
            pass
        elif option == 'Stochastic Dominance':
            pass
        else:
            pass
        return result

    def display(self):
        print(self)

    def __repr__(self):
        return "CSCV (Combinatorially Symmetric Cross Validation)"

    def __str__(self):
        return self.print()

    def plot(self):
        plt.plot(self.performance['IS'], self.performance['OOS'])
        plt.xlabel('IS '+self.measure)
        plt.ylabel('OOS '+self.measure)
        plt.title('Scatter Plot')


def calc_perf(ret, measure):
    if measure == 'avg return':
        return ret['ret'].mean()
    elif measure == 'sharpe ratio':
        return ret['ret'].mean()/ret['ret'].std()


def _test():
    data = get(utils.curpath() + '\..\\..\\data\\input.csv')
    data.reset_index(level=0, inplace=True)
    price = data.pivot_table(values='ADJPRC', index='DATE', columns='FIRMCO')

    """
    base_list = ['DATE', 'NAME', 'FNSECTCODE', 'FNSECTNAME', 'FIRMCO', 'RETM']
    main_list = ['MVmr', 'PERr', 'PBRr']
    sub_list = ['ROE4Qr', 'OPR4Qr']
    sector_list = ['FGSC.15','FGSC.20','FGSC.25','FGSC.30','FGSC.35','FGSC.45']
    port = utils.cleanse(data, selector=base_list + main_list+sub_list)

    def cut(port, SPP_cut=0.5, sub_cut=1):
        port['in_sector'] = port.FNSECTCODE.isin(sector_list)
        port['in_main'] = (port.MVmr + port.PERr + port.PBRr < SPP_cut)
        port['in_sub'] = (port.ROE4Qr + port.OPR4Qr > sub_cut)
        port['SPP'] = ((port['in_sector'] & port['in_main']) & port['in_sub'])
        return port.pivot_table(values='SPP', index='DATE', columns='FIRMCO').fillna(False)
    """

    """
    예제 1 : random select
    """
    N = 4
    strategy_set = [kf.Strategy('rand_'+str(i),
                                [kf.algos.RunMonthly(),
                                kf.algos.SelectRandomly(n=i+5),
                                kf.algos.WeighEqually(),
                                kf.algos.Rebalance()])
                    for i in range(N)]
    result = CSCV(strategy_set, prices=price, part=10)
    print(result)

    """
    예제 3

    strategy_set = []
    for i in range(20):
        for j in range(10):
            strategy_set.append(StrategyValue(SPP=j / 10, Profit=i / 10))

    result = CSCV(strategy_set, data, part=10)
    print(result)
    """


if __name__ == '__main__':
    _test()
