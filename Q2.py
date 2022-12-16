import numpy as np
import os
import matplotlib.pyplot as plt
from Q1 import read_files

script_dir = os.path.dirname(__file__)
rel_path = "Q1/"
abs_file_path = os.path.join(script_dir, rel_path)
output_dir = "Q2_results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# def f(x):
#     """Function to minimize. Test with Rosenbrock function"""
#     return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# def f_grad(x):
#     """Gradient of function to minimize. Test with Rosenbrock function."""
#     return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])

def f(A, b, c, x, t=1):
    """Function to minimize. Returns a scalar."""
    if (b-np.matmul(A, x) > 0).all():
        return t*np.dot(c, x) - np.sum(np.log(-np.matmul(A, x) + b))
    else:
        return np.inf

def f_grad(A, b, c, x, t=1):
    """Gradient of function to minimize. Returns a vector."""
    return t*c + np.dot(A.T, 1 / (-np.dot(A, x) + b))

def get_data():
    A, b = read_files("A3", "b3")
    m = len(b)
    n = len(A[0])
    # Two stacked identity matrices of size m x m
    A_tilde_r = np.vstack([-np.identity(m), -np.identity(m)])
    # Stack A on top of its negative
    A_tilde_l = np.vstack([A, -A])
    A_tilde = np.hstack([A_tilde_l, A_tilde_r])
    c_tilde = np.concatenate((np.zeros((n)), np.ones(m)), axis=0)
    b_tilde = np.concatenate((b, -b), axis=0)
    return A_tilde, b_tilde, c_tilde

def backtrack(epsilon):
    """Backtracking line search"""
    func_history = []
    grad_history = []
    alpha = 0.3
    beta = 0.5
    A_tilde, b_tilde, c_tilde = get_data()
    # Initialise x to value inside feasible region
    # x = c_tilde
    # Set x as psuedoinverse of A_tilde
    x = np.dot(np.linalg.pinv(A_tilde), b_tilde-5)
    grad = f_grad(A_tilde, b_tilde, c_tilde, x)
    func = f(A_tilde, b_tilde, c_tilde, x)
    iteration = 0

    while np.linalg.norm(grad, ord=2) > epsilon:
        t = 1   # Start t=1 for line search
        # First ensure in feasible region
        while f(A_tilde, b_tilde, c_tilde, x-t*grad) == np.inf:
            t = beta*t
        # Now satisfy backtracking condition
        while f(A_tilde, b_tilde, c_tilde, x-t*grad) > func - alpha*t*np.dot(grad, grad):
            t = beta*t
        x = x - t*grad
        iteration += 1
        # Next step size t
        grad = f_grad(A_tilde, b_tilde, c_tilde, x)
        # Next function value
        func = f(A_tilde, b_tilde, c_tilde, x)
        print(func)
        if iteration % 100 == 0:
            print("Iteration: ", iteration, "func", func)
        func_history.append(func)
        grad_history.append(np.linalg.norm(grad, ord=2))
    func_history = np.array(func_history)-func_history[-1]
    print(x, func)
    plot_func_history(func_history, "function")
    plot_func_history(grad_history, "gradient")

def plot_func_history(f, title):
    """Plot function history"""
    fig = plt.figure(figsize=(8,5))
    ax  = fig.add_subplot(111)
    ax.plot(f)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error in function value")
    ax.set_yscale('log')
    plt.savefig(output_dir+title+"_history.png")

backtrack(0.01)