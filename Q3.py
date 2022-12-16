import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
rel_path = "Q3/"
abs_file_path = os.path.join(script_dir, rel_path)
output_dir = "Q3_results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Read data
A_filepath = os.path.join(abs_file_path, 'A.csv')
x_filepath = os.path.join(abs_file_path, 'x0.csv')
A = np.genfromtxt(A_filepath, delimiter=',')
x0 = np.genfromtxt(x_filepath, delimiter=',')
b = np.matmul(A, x0)
m = len(b)
n = len(A[0])
# Lambda constant
lammy = 0.01*np.linalg.norm(2*np.matmul(A.T,b), ord=np.inf)

def f_original(X):
    """Barrier phi function to minimize. Returns a scalar."""
    x = X[:n]
    u = X[n:]
    if (u-x < 0).any() or (u+x < 0).any():
        return np.inf
    return np.linalg.norm(np.matmul(A, x) - b, ord=2) + lammy*np.sum(u)

def barrier(X):
    x = X[:n]
    u = X[n:]
    return - np.sum(np.log(u-x)) - np.sum(np.log(u+x))

def f(X, t):
    return t*f_original(X) + barrier(X)

def f_grad(X, t):
    """Gradient of function. Returns a vector where first m entries are wrt x and last m entries are wrt u."""
    x = X[:n]
    u = X[n:]
    f_grad_x = 2*t*np.matmul(A.T, np.matmul(A,x)-b) + 1/(u-x) - 1/(u+x)
    f_grad_u = t*lammy - 1/(u-x) - 1/(u+x)
    return np.concatenate((f_grad_x, f_grad_u))

def f_hess(X, t):
    """H_11. Returns a vector, which is the diagonals of the formal matrix."""
    x = X[:n]
    u = X[n:]
    # Vectors of diagonals of matrices
    H_11 = 2*t*np.matmul(A.T, A) + np.diag(1/np.square(u-x) + 1/np.square(u+x))
    H_12 = np.diag(-1/np.square(u-x) + 1/np.square(u+x))
    H_22 = np.diag(1/np.square(u-x) + 1/np.square(u+x))
    return np.block([[H_11, H_12], [H_12, H_22]])

def Newton(epsilon, t, X_0):
    """Newton's method for central path barrier.
    
    :param epsilon: tolerance for stopping condition
    :param t: central path parameter
    :param X_0: inital feasible point for X
    """
    # Inital feasible point for X
    X = X_0
    func = f(X, t)
    grad = f_grad(X, t)
    hess = f_hess(X, t)
    inv_hess = np.linalg.inv(hess)
    step_dir = -np.matmul(inv_hess, grad)
    nabla_f_grad = np.dot(grad.T, step_dir)
    alpha = 0.3 # defines comparison in backtracking line search
    beta = 0.5 # defines rate of decrease of tau, step size in each Newton iteration
    iteration = 0

    # Two-way stopping condition
    while 0.5*abs(np.dot(grad.T, step_dir)) > epsilon and np.linalg.norm(grad, ord=2) > epsilon:
        tau = 1 # Initial Newton step size, tuned with line search
        #  Check feasbility
        u = X[n:]
        x = X[:n]
        backtrack_it = 0
        while (u-x <= 0).any() or (u+x <= 0).any():
            Xdash = X + tau*step_dir
            u = Xdash[n:]
            x = Xdash[:n]
            tau = beta*tau
            backtrack_it += 1
            if backtrack_it > 100:
                print('Feasibility backtracking failed')
                break
        # Now satisfy backtracking condition
        backtrack_it = 0
        while f_original(X+tau*step_dir) > f_original(X) - alpha*tau*nabla_f_grad:
            tau = beta*tau
            backtrack_it += 1
            if backtrack_it > 500:
                print('Linesearch backtracking failed')
                break
        X = X + tau*step_dir
        func = f(X, t)
        grad = f_grad(X, t)
        hess = f_hess(X, t)
        inv_hess = np.linalg.inv(hess)
        step_dir = -np.matmul(inv_hess, grad)
        iteration += 1
        print(func)
    return X

def central_path(epsilon, t_init, t_max):
    t = t_init
    mu = 10 # defines rate of increase of t, central path parameter
    # Inital feasible point for X
    X = np.concatenate((np.ones(n)/2, np.ones(n)))
    while t < t_max:
        X = Newton(epsilon, t, X)
        t = mu*t  
    return X  

# Plot original and reconstructed signals
def plot(X, title):
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    ax.plot(X[:n])
    ax.set_ylabel('Signal component value')
    ax.set_xlabel('Component')
    plt.savefig(output_dir+title+"_signal.png")

X_0 = np.concatenate((np.ones(n)/2, np.ones(n)))
X = central_path(0.01, 0.1, 1e5)
print(X)

# X_0 = np.concatenate((np.zeros(n), np.ones(n)))
# X = Newton(0.01, 1e8, X_0)

plot(X, "Reconstructed")
plot(x0, "Original")