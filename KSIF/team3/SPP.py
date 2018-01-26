"""
SPP Strategy

"""
import KSIF as kf

__author__ = 'Seung Hyeon Yu'
__email__ = 'rambor12@business.kaist.ac.kr'



def value_signal(data, SPP_cut = 0.5, sub_cut=1):
    data['in_sector'] = data.FNSECTCODE.isin(sector_list)
    data['in_main'] = (data.MVmr + data.PERr + data.PBRr < SPP_cut)
    data['in_sub'] = (data.ROE4Qr + data.OPR4Qr > sub_cut)
    data['SPP'] = ((data['in_sector'] & data['in_main']) & data['in_sub'])
    return data.pivot_table(values='SPP', index='DATE', columns='FIRMCO').fillna(False)


#data = kf.get('data\\input.csv')


data = kf.get(r'C:\Users\ysh\Google 드라이브\package\KSIF\data\input.csv')
data.reset_index(level=0, inplace=True)
price = data.pivot_table(values='ADJPRC', index='DATE', columns='FIRMCO')

base_list = ['DATE', 'NAME', 'FNSECTCODE', 'FNSECTNAME', 'FIRMCO', 'RETM']
main_list = ['MVmr', 'PERr', 'PBRr']
sub_list = ['ROE4Qr', 'OPR4Qr']
sector_list = ['FGSC.15', 'FGSC.20', 'FGSC.25', 'FGSC.30', 'FGSC.35', 'FGSC.45']
data = kf.cleanse(data, selector=base_list + main_list + sub_list)
'''
s = kf.Strategy('value', [kf.algos.RunMonthly(),
                          kf.core.algos.SelectWhere(value_signal(port)),
                          kf.core.algos.WeighEqually(),
                          kf.core.algos.Rebalance()])

port = s.build_port(price)

t = kf.Backtest(s, price, start='2010-01-01')
res = kf.run(t)
res.display()
res.plot()
'''

# CSCV test
N = 3
strategy_set = [kf.Strategy('value_SPP=' + str(i) + '|sub='+str(j),
                            [kf.algos.RunMonthly(),
                             kf.core.algos.SelectWhere(value_signal(data, SPP_cut=(i+1)*0.2, sub_cut=(j+1)*0.3)),
                             kf.core.algos.WeighEqually(),
                             kf.core.algos.Rebalance()])
                for i in range(N) for j in range(N)]
strategy_set[0].build_port(price)
result = kf.CSCV(strategy_set, prices=price, part=10)
print(result)











# Experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S

dates = pd.date_range(start='2010-01-31', end='2010-03-31')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = 100

ret = pd.DataFrame({'DATE':dates})
ret = ret.join(pd.DataFrame({'A'+format(i+1,'02') : geometric_brownian_motion(T, N, sigma=0.1, S0=start_price) for i in range(20)}))
ret = ret.set_index('DATE')
ret.plot()
plt.show()




port = pd.DataFrame({})


# loop 1
dt = pd.to_datetime('2010-01-31')
temp = {'A01': 0.5, 'A04' : 0.3, 'A06':0.1, 'A09':0.2}

buf = pd.Series(temp).to_frame('w')
r_last = ret.ix[dt]
buf['DATE'] = dt


# loop 2
dt = pd.to_datetime('2010-02-28')
temp = {'A01': 0.3, 'A05' : 0.1, 'A08':0.1, 'A11':0.1, 'A20':0.4}

r_now = ret.ix[dt]
buf['r'] = r_now / r_last - 1
r_last = ret.ix[dt]
port = port.append(buf)

buf = pd.Series(temp).to_frame('w')
buf['DATE'] = dt


# loop 2
dt = pd.to_datetime('2010-03-31')
temp = {'A02': 0.6, 'A03' : 0.15, 'A04':0.1, 'A7':0.05, 'A17':0.1}

r_now = ret.ix[dt]
buf['r'] = r_now / r_last - 1
r_last = ret.ix[dt]
port = port.append(buf)

buf = pd.Series(temp).to_frame('w')
buf['DATE'] = dt








