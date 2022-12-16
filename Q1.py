from scipy.optimize import linprog
from scipy.linalg import lstsq
import numpy as np
import timeit
import os
import matplotlib.pyplot as plt
from scipy import stats

script_dir = os.path.dirname(__file__)
rel_path = "Q1/"
abs_file_path = os.path.join(script_dir, rel_path)
output_dir = "Q1_results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def read_files(A, b):
    # Read csv file into numpy array
    A_filepath = os.path.join(abs_file_path, A+'.csv')
    b_filepath = os.path.join(abs_file_path, b+'.csv')
    A = np.genfromtxt(A_filepath, delimiter=',')
    b = np.genfromtxt(b_filepath, delimiter=',')
    return A, b

def solve_l1():
    """l1 norm approximation"""
    results = []
    runtimes = []
    ns = []
    for i in range(1, 6):
        a = 'A'+str(i)
        b = 'b'+str(i)
        A, b = read_files(a, b)
        m = len(b)
        n = len(A[0])
        ns.append(n)
        # Two stacked identity matrices of size m x m
        A_tilde_r = np.vstack([-np.identity(m), -np.identity(m)])
        # Stack A on top of its negative
        A_tilde_l = np.vstack([A, -A])
        A_tilde = np.hstack([A_tilde_l, A_tilde_r])
        c_tilde = np.concatenate((np.zeros((n)), np.ones(m)), axis=0)
        b_tilde = np.concatenate((b, -b), axis=0)

        start_time = timeit.default_timer()
        x = linprog(c_tilde, A_tilde, b_tilde, method='highs-ipm').x
        runtimes.append(timeit.default_timer() - start_time)
        residual = np.dot(A, x[:n]) - b
        results.append(np.sum(np.abs(residual)))
        if i == 5:
            plot_residual(residual, 'l_1', dist=True, type=1)
        if i == 3:
            # save as txt
            np.savetxt(output_dir+'l1_case3.csv', x, delimiter=',')
    plot_runtime(ns, runtimes, 'l_1')
    plot_results(ns, results, 'l_1')
    print(results)
    print(runtimes)

def solve_l2():
    """least squares approximation"""
    results = []
    runtimes = []
    ns = []
    for i in range(1, 6):
        a = 'A'+str(i)
        b = 'b'+str(i)
        A, b = read_files(a, b)
        ns.append(len(A[0]))
        start_time = timeit.default_timer()
        x = lstsq(A, b)[0] # first return element is x array (see scipy docs)
        runtimes.append(timeit.default_timer() - start_time)
        residual = np.dot(A, x) - b
        results.append(np.linalg.norm(residual, 2))
        if i == 5:
            plot_residual(residual, 'l_2', dist=True, type=2)
    plot_runtime(ns, runtimes, 'l_2')
    plot_results(ns, results, 'l_2')
    print(results)
    print(runtimes)

def solve_linf():
    """l_inf norm approximation"""
    results = []
    runtimes = []
    ns = []
    for i in range(3, 4):
        a = 'A'+str(i)
        b = 'b'+str(i)
        A, b = read_files(a, b)
        m = len(b)
        n = len(A[0])
        ns.append(n)
        # Stack A on top of its negative
        A_tilde_l = np.vstack([A, -A])
        # Add a column of -1s
        A_tilde = np.hstack([A_tilde_l, -np.ones((2*m, 1))])
        c_tilde = np.concatenate((np.zeros((n)), np.ones(1)), axis=0)
        b_tilde = np.concatenate((b, -b), axis=0)

        start_time = timeit.default_timer()
        x = linprog(c_tilde, A_tilde, b_tilde, method='highs-ipm').x
        runtimes.append(timeit.default_timer() - start_time)
        residual = np.dot(A, x[:n]) - b
        results.append(np.max(np.abs(residual)))
        if i == 5:
            plot_residual(residual, 'l_inf')
    plot_runtime(ns, runtimes, 'l_inf')
    plot_results(ns, results, 'l_inf')
    print(results)
    print(runtimes)

def plot_residual(x, title, dist=False, type=0):
    """Plot residual distribution histogram"""
    fig = plt.figure(figsize=(8,2))
    ax  = fig.add_subplot(111)
    n, bins, _ = ax.hist(x, bins=60, density=1)
    if dist == True and type == 2:
        mean, std = stats.norm.fit(x)
        p = stats.norm.pdf(bins, mean, std)
        ax.plot(bins, p, 'k', linewidth=2)
        print(mean, std)
    if dist == True and type == 1:
        # Fit Laplacian distribution to data
        mu, b = stats.laplace.fit(x)
        p = stats.laplace.pdf(bins, mu, b)
        ax.plot(bins, p, 'k', linewidth=2)
        print(mu, b)
    ax.set_title(title)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Relative normalised occurrence')
    plt.savefig(output_dir+title+'.png')

def plot_runtime(n, runtimes, title):
    """Plot runtime vs n"""
    fig = plt.figure(figsize=(5,3))
    ax  = fig.add_subplot(111)
    ax.plot(n, runtimes)
    ax.set_xlabel('n')
    ax.set_ylabel('Runtime (s)')
    plt.savefig(output_dir+title+'_runtime_sm.png')

def plot_results(n, results, title):
    """Plot residual vs n for all three norms"""
    fig = plt.figure(figsize=(10,6))
    ax  = fig.add_subplot(111)
    ax.plot(n, results, label='l_1')
    ax.set_xlabel('n')
    ax.set_ylabel('Residual')
    ax.legend()
    plt.savefig(output_dir+title+'_results.png')

if __name__ == '__main__':
    solve_l1()
    # solve_linf()
    # solve_l2()