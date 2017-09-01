# Import Packages
import numpy as np
import scipy.optimize as opt
import SS_6 as ss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

def get_TPI(tpi_params, b_init, TPI_graphs):
    S, T, beta, sigma, nvec, A, alpha, delta, b_ss, xi, TPI_tol, EulDiff = tpi_params
    L = np.sum(nvec)
    Kvec = np.linspace(np.sum(b_init), np.sum(b_ss), T)
    err = 100
    iteration = 0
    while err > TPI_tol:
        rvec = ss.get_r(Kvec, L, alpha, A, delta)
        wvec = ss.get_w(Kvec, L, alpha, A)
        bmatrix = np.zeros([T,S-1])
        bmatrix[0,:] = b_init

        rows = np.zeros([T-S+1,S-1])

        for i in range(S-3,-1,-1):
            binit = b_init[i]
            periods = np.diag(bmatrix,k=i).shape[0]
            steady_state = False
            pass_args=(delta, alpha, beta, sigma, A, nvec[i+1:],
                       rvec[:periods], wvec[:periods], steady_state, EulDiff, binit)
            guess = np.diag(bmatrix,k=i+1).flatten()
            b_result = opt.fsolve(ss.EulErr, guess, args=(pass_args))
            b_result = np.array([b_result])
            b_result = np.append([0],b_result)
            bmat_new = np.diagflat(b_result, i)
            bmat_new = np.vstack((bmat_new,rows))
            bmatrix = bmatrix+bmat_new


        K_ss = np.sum(b_ss)
        r_ss = ss.get_r(K_ss, L, alpha, A, delta)
        w_ss = ss.get_w(K_ss, L, alpha, A)
        rvec = np.append(rvec, np.ones(S-2)*r_ss)
        wvec = np.append(wvec, np.ones(S-2)*w_ss)

        for i in range(-1,-T,-1):
            binit = 0
            periods = np.diag(bmatrix,k=i).shape[0]
            steady_state = False
            pass_args=(delta, alpha, beta, sigma, A, nvec,
                       rvec[-i-1:S-i-1], wvec[-i-1:S-i-1], steady_state, EulDiff, binit)
            if i==-1:
                guess = np.diag(bmatrix,k=i+1).flatten()
            else:
                gues  =b_result
            b_result = opt.fsolve(ss.EulErr, guess, args=(pass_args))
            b_result = np.array(b_result)
            bmat_new = np.diagflat(b_result[:periods], periods-S+1)
            if periods==(S-1):
                rows_above = np.zeros([-i,S-1])
                rows_below = np.zeros([T-S+1+i,S-1])
                bmat_new = np.vstack((rows_above,bmat_new,rows_below))
            else:
                bmat_new = np.vstack((rows,bmat_new))
            bmatrix = bmatrix+bmat_new

        iteration += 1
        Knew = np.sum(bmatrix,axis=1)
        err = np.sum(np.square(Kvec-Knew))
        Kvec = xi*Knew + (1-xi)*Kvec
        print("Iteration: ", iteration, "   Error: ", err)

    if TPI_graphs:
        xvals = np.arange(T)
        fig, ax = plt.subplots()
        plt.plot(xvals, Kvec)
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Transition Path for Capital Stock', fontsize=20)
        plt.xlabel('Period')
        plt.ylabel('Capital Stock')
        plt.savefig('K_TPI.png')
        plt.close
    return Kvec
