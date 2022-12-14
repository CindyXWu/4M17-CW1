from scipy.optimize import linprog
import numpy as np
import timeit
import os, sys

script_dir = os.path.dirname(__file__)
rel_path = "Q1/"
abs_file_path = os.path.join(script_dir, rel_path)

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
    for i in range(1, 6):
        a = 'A'+str(i)
        b = 'b'+str(i)
        A, b = read_files(a, b)
        m = len(b)
        n = len(A[0])
        # Two stacked identity matrices of size m x m
        A_tilde_r = np.concatenate( (-np.identity(m), -np.identity(m)) , axis=0)
        # Stack A on top of its negative
        A_tilde_l = np.concatenate( (A, -A), axis=0)
        # Form final A_tilde
        A_tilde = np.concatenate( (A_tilde_l, A_tilde_r), axis=1)
        c_tilde = np.concatenate( (np.zeros((n, 1)), np.ones((m, 1)) ), axis = 0)
        b_tilde = np.concatenate( (b, -b), axis = 0)

        start_time = timeit.default_timer()
        x = linprog(c_tilde, A_tilde, b_tilde).x
        runtimes.append(timeit.default_timer() - start_time)
        residual = np.abs(np.dot(A, x[:n]) - b)
        results.append(np.sum(residual))
    print(results)
    print(runtimes)

def solve_linf():
    """l_inf norm approximation"""
    results = []
    runtimes = []
    for i in range(1, 6):
        a = 'A'+str(i)
        b = 'b'+str(i)
        A, b = read_files(a, b)
        m = len(b)
        n = len(A[0])
        # Stack A on top of its negative
        A_tilde = np.concatenate( (A, -A), axis=0)
        # Add a column of -1s
        A_tilde = np.concatenate( (A_tilde, -np.ones((2*m, 1)) ), axis=1)
        c_tilde = np.concatenate( (np.zeros((n)), np.ones(1) ), axis = 0)
        b_tilde = np.concatenate( (b, -b), axis = 0)

        start_time = timeit.default_timer()
        x = linprog(c_tilde, A_tilde, b_tilde).x
        runtimes.append(timeit.default_timer() - start_time)
        residual = np.abs(np.dot(A, x[:n]) - b)
        results.append(np.max(residual))
    print(results)
    print(runtimes)

# solve_l1()
solve_linf()