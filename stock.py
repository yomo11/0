#basic
import numpy as np
import pandas as pd

#get data
import pandas_datareader as pdr

#visual
import matplotlib.pyplot as plt
import mpl_finance as mpf
#matplotlib inline
import seaborn as sns

#time
import datetime as datetime

#talib
import talib

#analysis
#import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sympy import *

#get date
date_end = input("input date(yyyy-mm-dd): ")
d = datetime.datetime.strptime(date_end, '%Y-%m-%d')
date_start = d - datetime.timedelta(days = 15)
date_start = str(date_start)

#get stock code
stock_code = input("input stock code(XXXX)): ")
stock_code += ".TW"

#get stock`s close price in 15days 
stock = pdr.DataReader(stock_code, 'yahoo', start=date_start, end=date_end)

#test print
t = stock['Close']
print(t)
t1 = t[0]
print(t1)
'''
t1 = stock.loc[2:16,0]
print(t1)
'''
#stock analysis
stock_pd = pd.Series(t)
#var = stock_pd.var()
#print(var)

'''
def linear_regression(X, Y):
	a, b = symbols('a b')
	residual = 0

	for i in range(15):
		residual += (Y[i] - (a * X[i] + b) ) ** 2
	
	print (expand(residual))
	f1 = diff(residual, a)
	f2 = diff(residual, b)
	print(f1)
	print(f2)

	res = solve([f1, f2], [a, b])
	return res[a], res[b]

a, b = linear_regression(stock['Close'], stock[:,0])
LR_X = stock['Close']
h = lambda x: a * x + b
H = np.vectorize(h)
LR_Y = H(LR_X)

plt.plot(LR_X, LR_Y, 'g')
plt.plot(stock['Close'], stock[:,0], 'ro')
plt.show()
'''	
