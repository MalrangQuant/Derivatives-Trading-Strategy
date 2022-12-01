import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from warnings import filterwarnings
filterwarnings('ignore')

class delta_hedging:

    def __init__(self, input):
 
        self.s0         = input['s0']               # 현재가격
        self.k          = input['k']                # 행사가격
        self.rf         = input['rf']               # 무위험이자율
        self.sigma      = input['sigma']            # 변동성
        self.mu         = input['mu']               # 평균수익률
        self.T          = input['T']                # 투자기간
        self.q          = input['q']                # 배당금 
        self.iv         = input['imvol']            # 내재변동성

        self.interval   = input['interval']         # T의 기간 텀
        self.time_step  = input['time_step']        # Investment Horizon
        self.simul      = input['simul_num']        # 시뮬레이션 횟수
        self.num_shares = input['num_shares']       # 종목 수

        if self.interval == 'weekly':               
            self.dt = self.T / 52                   
        elif self.interval == 'daily':
            self.dt = self.T / 255
        else:
            self.dt = self.interval

        self.tau        = self.time_step * self.dt
        self.tau_arr    = (np.flip(np.arange(0, self.time_step+1)) * self.dt).reshape(-1,1)

    def bs_call(self, PutCall):
        '''PutCall for 'c', 'p' '''
        d1 = (np.log(self.s0/self.k) +((self.rf - self.q +0.5*np.power(self.sigma, 2))*self.tau))/(self.sigma*np.sqrt(self.tau))
        d2 = d1 - self.sigma*np.sqrt(self.tau)
        value = self.s0*norm.cdf(d1) - self.k*np.exp(-self.rf*self.tau)*norm.cdf(d2)
        return value

    def stock_path(self):

        value = np.ones((2, self.simul)) * self.s0

        for idx in range(1, self.T * self.time_step + 1):
            d_term = (self.mu - self.q - 0.5 * self.sigma**2) * self.dt
            s_term = self.sigma * np.sqrt(self.dt) * np.random.normal(size=(self.simul))

            value = np.vstack([value, value[-1] * np.exp(d_term + s_term)])
            
        value = value[1:]

        return value


    def d1_cal (self, s, k, r, q, vol, t):
        return (np.log(s/k) +((r-q+0.5*np.power(vol, 2))*t))/(vol*np.sqrt(t))

    def path_delta(self, stock_path=None):
        
        if stock_path is not None:
            path = stock_path
        else:
            path = self.stock_path()

        delta = norm.cdf(self.d1_cal(path, self.k, self.rf, self.q, self.iv, self.tau_arr))
        
        return delta

    def path_changes (self, path_1):
        '''calculate path's changes'''

        return np.vstack([path_1[0], path_1[1:] - path_1[:-1]])

    def path_delta_shares(self, stock_path=None):
        '''input stock path (stock price path)'''

        if stock_path is not None:
            path = self.path_delta(stock_path)
        else:
            stock_path = self.stock_path()
            path = self.path_delta(stock_path)
        
        return path * self.num_shares

    def path_delta_shares_cost (self, stock_path=None, tr=0):
        '''input stock path (stock price path)'''

        if stock_path is not None:
            path = self.path_delta_shares(stock_path)
        else:
            stock_path = self.stock_path()
            path = self.path_delta_shares(stock_path)

        cost = path * stock_path
        cost = np.where(cost>0, cost*(1+tr), cost)

        return cost

    def path_delta_hedged_cum (self, stock_path=None, tr=0):

        if stock_path is not None:
            path = self.path_delta_shares_cost(stock_path, tr)

        else:
            stock_path = self.stock_path()
            path = self.path_delta_shares_cost(stock_path, tr)

        path_cost_ch = self.path_changes(path)

        delta_path_cumcost = np.repeat(path_cost_ch[0], stock_path.shape[0]).reshape(-1, stock_path.shape[1])
        upper = (1 + self.rf*1/52)

        for x in range(0, stock_path.shape[1]):
            for idx in range(stock_path[:,x].shape[0]-1):
                delta_path_cumcost[idx+1, x] = np.round(delta_path_cumcost[idx,x] * upper + path_cost_ch[idx+1, x])

        return delta_path_cumcost

    def delta_hedging_int_cost (self, stock_path=None, tr=0):

        if stock_path is not None:
            path = self.path_delta_hedged_cum(stock_path, tr)
        else:
            stock_path = self.stock_path()
            path = self.path_delta_hedged_cum(stock_path, tr)

        return path * (self.dt * self.rf)

    def hedging_cost(self, stock_path=None, tr=0):

        if stock_path is not None:
            delta = self.path_delta(stock_path)
        else:
            stock_path = self.stock_path()
            delta = self.path_delta(stock_path)

        path_arr = np.vstack([delta[0], (delta[1:] - delta[:-1])])
        
        arr = (path_arr * stock_path)
        arr = np.where(arr>0, arr*(1+tr), arr)

        hedge_arr = ( arr * np.exp(self.rf * self.tau_arr)).cumsum(axis=0)[-1]
        hedge_arr = np.where(hedge_arr > self.k, hedge_arr - self.k, hedge_arr)

        return hedge_arr
            
