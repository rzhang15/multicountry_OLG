# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def get_nvec(S):
    cutoff = int(round(2*S/3))
    nvec1 = np.ones(cutoff)
    nvec2 = np.ones(S-cutoff)*0.2
    nvec = np.concatenate((nvec1,nvec2))
    return nvec

def get_r(K, L, alpha, A, delta):
    return alpha*A*(L/K)**(1-alpha)-delta

def get_w(K, L, alpha, A):
    return (1-alpha)*A*((K/L)**alpha)

def get_cvec(bvec, nvec, r, w, binit):
    b2 = np.append([binit], bvec)
    b3 = np.append(bvec, 0)
    cvec = (1 + r) * b2 + w * nvec - b3
    return (1+r)*b2+w-b3

def get_MUvec(cvec, sigma):
    epsilon = 0.0001
    b2 = (-sigma*(epsilon**(-sigma-1)))/2
    b1 = (epsilon**(-sigma))-2*b2*epsilon
    MU_vec = np.zeros_like(cvec)
    MU_vec[cvec<epsilon] = 2*b2*cvec[cvec<epsilon]+b1
    MU_vec[cvec>=epsilon] = cvec[cvec>=epsilon]**(-sigma)
    return MU_vec

def EulErr(bvec, *args):
    delta, alpha, beta, sigma, A, nvec, rvec, wvec, steady_state, EulDiff, binit = args
    S = len(nvec)
    if steady_state:
        K = bvec.sum()
        L = nvec.sum()
        rvec = np.ones(S)*get_r(K, L, alpha, A, delta)
        wvec = np.ones(S)*get_w(K, L, alpha, A)
    MU_vec = get_MUvec(get_cvec(bvec, nvec, rvec, nvec*wvec, binit), sigma)
    if EulDiff:
        err_vec = MU_vec[:-1] - beta*(1+rvec[1:])*MU_vec[1:]
    else:
        err_vec = (beta*(1+rvec[1:])*MU_vec[1:]/MU_vec[:-1])-1
    return err_vec

def get_SS(ss_params, bvec_guess, SS_graphs):
    beta, sigma, nvec, A, alpha, delta, SS_tol, EulDiff = ss_params
    steady_state = True
    rvec = 0
    wvec = 0
    binit = 0
    eul_args = (delta, alpha, beta, sigma, A, nvec, rvec, wvec, steady_state, EulDiff, binit)
    b_ss = opt.fsolve(EulErr, bvec_guess, args=(eul_args), xtol=SS_tol)
    K = b_ss.sum()
    L = np.sum(nvec)
    w = get_w(K, L, alpha, A)
    r = get_r(K, L, alpha, A, delta)
    wvec = w*np.ones(len(nvec))
    rvec = r*np.ones(len(nvec))
    c_ss = get_cvec(b_ss, nvec, rvec, nvec*wvec, binit)
    print('Savings: ', b_ss)
    print('Consumption: ', c_ss)
    print('Wage: ', w)
    print('Interest: ', r, '\n')
    if SS_graphs:
        xvals = np.arange(1,len(wvec)+1,1)
        fig, ax = plt.subplots()
        plt.plot(xvals[1:], b_ss)
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-State Savings (80 Periods)', fontsize=20)
        plt.xlabel('Age')
        plt.ylabel('Savings')
        plt.savefig('stead_state_savings_80.png')
        plt.close

        fig, ax = plt.subplots()
        plt.plot(xvals, c_ss)
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-State Consumption (80 Periods)', fontsize=20)
        plt.xlabel('Age')
        plt.ylabel('Consumption')
        plt.savefig('stead_state_consumption_80.png')
        plt.close
    return b_ss
