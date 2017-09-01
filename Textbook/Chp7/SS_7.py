# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def MU_cfe(n, theta):
    return n**(1/theta)

def MU_elp(n, b, l, nu):
    constant = b/(l**nu)
    bracket1 = (1-(n/l)**nu)**(1/nu-1)
    return constant*bracket1*(n**(nu-1))

def find_ellipse(params, *args):
    b = params[0]
    nu = params[1]
    theta, l, n_vec = args
    errvec = np.sum(np.square(MU_cfe(n_vec, theta)-MU_elp(n_vec, b, l, nu)))
    return errvec

def fit_ellip(theta, l):
    labor = np.linspace(0.05,0.85,1000)
    guess = (0.2, 0.2)
    pass_args = theta, l, labor
    parameters = opt.minimze(find_ellipse, guess, args=(pass_args))
    return parameters.x
