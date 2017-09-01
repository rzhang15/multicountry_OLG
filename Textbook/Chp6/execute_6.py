# Import packages
import numpy as np
import SS_6 as ss
import TPI_6 as tpi

# Household parameters
life = int(80)
S = int(80)
beta_annual = .96
beta = beta_annual ** (life/S)
sigma = 3.0
nvec = ss.get_nvec(S)
L = nvec.sum()
# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1 - delta_annual) ** (life/S))
# SS parameters
SS_tol = 1e-13
SS_graphs = False
# TPI parameters
T = 150
TPI_solve = True
TPI_tol = 1e-9
xi = 0.9
TPI_graphs = True
# Overall parameters
EulDiff = False

print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION')
bvec_guess = np.ones(S-1)*0.1
ss_params = (beta, sigma, nvec, A, alpha, delta, SS_tol, EulDiff)
ss_output = ss.get_SS(ss_params, bvec_guess, SS_graphs)

if TPI_solve:
    print('BEGIN EQUILIBRIUM TIME PATH COMPUTATION')
    b_ss = ss_output
    b_init = 0.93*b_ss
    tpi_params = (S, T, beta, sigma, nvec, A, alpha, delta, b_ss, xi, TPI_tol, EulDiff)
    tpi_output = tpi.get_TPI(tpi_params, b_init, TPI_graphs)
