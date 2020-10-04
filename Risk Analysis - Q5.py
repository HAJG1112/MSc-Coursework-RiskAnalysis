
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

class RiskAnalysis():
    def __init__(self ,S0 ,K ,T, horizon ,vol ,predicted_days, r, M):
        self.S0 = S0
        self.K = K
        self.T = T
        self.vol = vol
        self.predicted_days = predicted_days
        self.horizon = horizon
        self.r = r
        self.M = M
        self.dt = self.predicted_days/250

                #### These are the set up functions ###

###### Computes the intital value of the call with respect to the parameters initalized
    def initial_call(self):
        d1 = (np.log(self.S0 / self.K) + ((self.r + (self.vol ** 2) / 2) * self.T)) / (self.vol * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + ((self.r - (self.vol ** 2) / 2) * self.T)) / (self.vol * np.sqrt(self.T))
        call = self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        return call

###### Black scholes function which returns inputed values
    def Black_Scholes(self, S, K, r, vol, T):
        d1 = (np.log(S/K) + (r + (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
        d2 = (np.log(S/K) + (r - (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

####### Returns the intial value of theta based on the closed form solution of the Black scholes equation
    def call_theta(self):
        d1 = self.d1(self.S0, self.K, self.r, self.vol, self.T)
        d2 = self.d2(self.S0, self.K, self.r, self.vol, self.T)
        x = -(self.S0*norm.pdf(d1)*self.vol)/(2*np.sqrt(self.T))
        y = -(self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(d2))
        theta = x + y
        return theta

####### Returns the intial value of delta based on the closed form solution of the Black scholes equation
    def call_delta(self):
        d1 = (np.log(self.S0/self.K) + ((self.r + self.vol**2/2)*self.T))/self.vol*np.sqrt(self.T)
        delta_call = norm.cdf(d1)
        return delta_call

####### Returns the intial value of Gamma based on the closed form solution of the Black scholes equation
    def call_gamma(self):
        d1 = (np.log(self.S0/self.K) + ((self.r + self.vol**2/2)*self.T))/self.vol*np.sqrt(self.T)
        n_d1 = norm.pdf(d1)
        gamma = n_d1/(self.S0*self.vol*np.sqrt(self.T))
        return gamma

####### Empty function to compute d1
    def d1(self, s, k, r, vol, T):
        d1 = (np.log(s / k) + (r + (vol ** 2) / 2) * T) / vol * np.sqrt(T)
        return d1
###### empty function to compute d2
    def d2(self, s, k , r, vol, T):
        d2 = (np.log(s / k) + (r - (vol ** 2) / 2) * T) / vol * np.sqrt(T)
        return d2

                             #### Returns the calculations for the Exact VaR approach ####

#### This function returns the Exact VaR according to the 99% confidence interval
    def VaR_exact(self):
        call_t0 = self.initial_call()
        worst = self.S0 * np.exp(-2.33 * self.vol * self.horizon)
        call_value_worst = self.Black_Scholes(worst, self.K, self.r, self.vol, self.T - (self.horizon**2))  #value of call with worst case price and short horizon (T - ndt)
        E_VaR = call_t0 - call_value_worst    #inital call price subtract worst case price
        return E_VaR

### This function returns the delta approximate according to the Exact VaR calculation
    def delta_approx(self):
        worst = self.S0 * np.exp(-2.33 * self.vol * self.horizon)
        r_t = np.log(worst/self.S0)    #worst case return
        dP_P = np.exp(r_t)-1         #change in price in linear terms
        delta_c = self.call_delta()
        dV = (self.call_theta()*(self.horizon**2))+(delta_c * self.S0 * dP_P)  #value change in call option
        return -dV

### function returns the delta-gamma approx according to the Exact VaR calculation
    def delta_gamma_approx(self):
        worst = self.S0 * np.exp(-2.33 * self.vol * self.horizon)
        dS = np.log(worst / self.S0)  # log returns are i.i.d
        r_t = np.exp(dS) - 1
        theta_c = (self.call_theta()*self.horizon**2)
        delta_c = (self.call_delta() * self.S0*r_t)
        gamma_c = (self.call_gamma() * 0.5 * (self.S0**2)*(r_t**2))
        dV = theta_c + delta_c + gamma_c
        return -dV

                                #### Returns MCS approach calculations ####

    def var_computation_linear(self):
        linear = np.zeros((self.M, 1), float)
        #simulation loop
        for i in range(0,M):
            r_t = np.random.normal(0,1) * self.vol * self.horizon  #return = vol*root(n-days)*epsilon
            dP_P = np.exp(r_t) - 1
            linear[i] = self.call_theta()*self.horizon**2 + self.call_delta()*self.S0*dP_P
        linear = pd.DataFrame(linear)
        linear.columns = ['dV']
        return -linear   #returns all changes in the price of the option (M x 1) vectpr

    def var_computation_quadratic(self):
        quad = np.zeros((self.M, 1), float)
        #simulation loop
        for i in range(0,M):
            r_t = np.random.normal(0,1) * self.vol * self.horizon
            dP_P = np.exp(r_t) - 1
            quad[i] = self.call_theta()*self.horizon**2 + self.call_delta()*self.S0*dP_P + 0.5*self.call_gamma()*(self.S0**2)*(dP_P**2)
        quad = pd.DataFrame(quad)
        quad.columns = ['dV']
        return -quad       #returns all changes in the price of the option (M x 1) vectpr

    def var_revaluation(self):
        reval= np.zeros((self.M, 1), float)
        for i in range(0, M):
            r_t = np.random.normal(0, 1) * self.vol * self.horizon
            dP_P = np.exp(r_t) - 1
            reval[i] = self.Black_Scholes(self.S0*(1+dP_P), self.K, self.r, self.vol ,self.T - self.horizon**2) - self.initial_call()
        reval = pd.DataFrame(reval)
        reval.columns = ['dV']
        return -reval       #returns all changes in the price of the option (M x 1) vectpr

######################################################################################################################
### DO NOT CHANGE ANYTHING ABOVE ###

S0 = 100
K = 100
T = 1
vol = 0.3     #this is p.a volatility
predicted_days = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
r = 0  # continuously compounded
M = 10000  # number of simulations to run

for i in range(len(predicted_days)):
    print('horizon length = ', predicted_days[i])
    horizon = np.sqrt(predicted_days[i] / 250)
    ra = RiskAnalysis(S0, K, T, horizon, vol, predicted_days[i], r, M)
    print('THE EXACT APPROACH')
    print('exact var', ra.VaR_exact())
    print('delta approx exact' , ra.delta_approx())
    print('delta gamma approx exact', ra.delta_gamma_approx())
    print('THE MCS APPROACH')
    print('var linear', np.quantile(ra.var_computation_linear(),0.01))   #print 1% quantile of the linear vector
    print('var quadratic', np.quantile(ra.var_computation_quadratic(), 0.01)) #print 1% quantile of the quadratic vector
    print('var revaluation', np.quantile(ra.var_revaluation(), 0.01))   #print 1% quantile of the Full revaluation vector
    print('\n')

######################################### Density plots ########################################
#dist = pd.DataFrame(ra.var_revaluation())
#fig, ax = plt.subplots()
#dist.plot.kde(ax=ax, legend=False, title='Simulated Revaluated PnL - X Day Horizon')
#dist.plot.hist(density=True, ax=ax)
#ax.set_ylabel('Probability')
#ax.set_xlabel('PnL of the Call Option')
#ax.grid(axis='y')
#ax.set_facecolor('#eafff5')
#plt.show()


