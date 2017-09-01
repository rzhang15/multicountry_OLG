# Import packages
import numpy as np
import SS_7 as ss
import TPI_7 as tpi

# Household parameters
life = int(80)
S = int(80)
beta_annual = .96
beta = beta_annual ** (life/S)
sigma = 3.0
l_tilde = 1
theta = 2.0
MU_l_graphs = True
# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1 - delta_annual) ** (life/S))
# SS parameters
SS_solve = False
SS_tol = 1e-13
SS_graphs = False
# TPI parameters
T = 100
TPI_solve = False
TPI_tol = 1e-9
xi = 0.9
TPI_graphs = True
# Overall parameters
EulDiff = False

b_ellip, upsilon = ss.fit_ellip(theta, l_tilde)

if MU_l_graphs:
    xvals = np.linspace(0.05,0.85,1000)
    fig, ax = plt.subplots()
    plt.plot(xvals, ss.MU_cfe(xvals, theta))
    plt.plot(xvals, ss.MU_elp(xvals, b_ellip, l_tilde, upsilon))
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Marginal Utilities of Labor', fontsize=20)
    plt.xlabel('Leisure')
    plt.ylabel('Marginal Utility of Labor')
    plt.savefig('MU_labor.png')
    plt.show()
    plt.close

if SS_solve:
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
