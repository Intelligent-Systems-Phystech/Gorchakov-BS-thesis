# Import namespaces
import math
import numpy as np
import scipy.integrate as integrate

from scipy.stats import norm

from tqdm.notebook import tqdm, trange

from numba import njit


def joint_cc_gt(x, x_An, x_bn, controllable_idxs = None):
    """
    Returns the probability of satisfying joint linear constraint under probability
    \sum_{i=1}^m \log \Phi \left(- (a_i^\top x - d_i) \right)
    
    args:
        x(n,) array-like floats: control variable (optimization variable)
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
    
    """
    if controllable_idxs is None:
        vec = np.dot(x_An, x) - x_bn
    else:
        vec = np.dot(x_An[:, controllable_idxs], x[controllable_idxs]) - x_bn[controllable_idxs]
    #mean = np.max(vec)
    #return -mean
    return np.sum(np.array([np.log(norm.cdf(-v)) for v in vec]))

def joint_cc_grad(x, x_An, x_bn, eta, eta_var=True):
    """return the gradient of 
        joint chance constraint constraint at (x, eta) or x, depends on eta_var
    Args:
        x (ndarray(n,)): point (e.g., generations)
        x_An (ndarray(m,n)): matrix of linear constraints
        x_bn (ndarray(m,)): vector of R.H.S.s
        eta (float): relibaility parameters
        eta_var (bool, optional): if include gradient on eta into the returning value. Defaults to True.

    Returns:
        ndarray(n,) ndarray(n+2,): grad on x or x and eta_1, eta_2
    """
    pdf_cdf_ratio = norm.pdf(x_bn - np.dot(x_An, x)) / norm.cdf(x_bn - np.dot(x_An, x))
    sum_term_grad_x = pdf_cdf_ratio.reshape(-1, 1) * x_An
    grad_x = sum_term_grad_x.sum(axis=0)
    grad_eta = np.array([-1 / (1. - eta), 0])
    if eta_var:
        full_grad = np.hstack((grad_x, grad_eta))
        return full_grad
    else:
        return grad_x
def objective(c, x):
    """
    Returns the value of objective function from initial optimization problem
    c^\top x
    args:
        x(n,) array-like floats: control variable (optimization variable)
        c(n,) array-like floats: vector of costs c
    """
    return np.dot(c, x)
def objective_barrier(c, x, constr_foo, r):
    """
    Returns the value of penalized objective function. Penalization ensures the constraint satistaction
    c^\top x + r * max(0, constr_foo(x))**2
    args:
        x(n,) array-like floats: control variable (optimization variable)
        c(n,) array-like floats: vector of costs c
        constr_foo(x) function: g(x) from a constraint of form g(x) <= 0
        r float: penalization parameter
    """
    return np.dot(c, x[:len(c)]) + r * np.max((0.,constr_foo(x))) ** 2

def objective_barrier_multiple(c, x, constr_foos, r):
    """
    Returns the value of penalized objective function. Penalization ensures the constraint satistaction
    c^\top x + r * max(0, constr_foo(x))**2
    args:
        x(n,) array-like floats: control variable (optimization variable)
        c(n,) array-like floats: vector of costs c
        constr_foo(x) function: g(x) from a constraint of form g(x) <= 0
        r float: penalization parameter
    """
    return np.dot(c, x[:len(c)]) + r * sum([np.max((0.,constr_foo(x))) ** 2 for constr_foo in constr_foos])

def objective_barrier_multiple_grad(c, x, constr_foos, constr_foos_grads, r, eta_var=True):
    """
    Returns the gradient of penalized objective function at x.
    args:
        x(n,) array-like floats: control variable (optimization variable)
        c(n,) array-like floats: vector of costs c
        constr_foo(x) function: g(x) from a constraint of form g(x) <= 0
        r float: penalization parameter
    """
    if eta_var:
        return np.hstack((c, np.array([0.,0.]))) + 2 * r * sum([np.max((0.,constr_foo(x))) * constr_foo_grad(x) for constr_foo, constr_foo_grad in zip(constr_foos, constr_foos_grads)])
    else:
        return c + 2 * r * sum([np.max((0.,constr_foo(x))) * constr_foo_grad(x) for constr_foo, constr_foo_grad in zip(constr_foos, constr_foos_grads)])

def check_cc(x, x_An, x_bn, controllable_idxs = None):
    """
    Returns the value of chance constraint at current point x
    args:
        x(n,) array-like floats: control variable (optimization variable)
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
    """
    return np.exp(joint_cc_gt(x, x_An, x_bn, controllable_idxs = controllable_idxs))
def check_cc_scenario(x, x_An, x_bn, nsmp=10000, cov_ineqs=None):
    """
    Returns empirical estimate of chance constraint at point x based on nsmp Monte-Carlo samples
    args:
        x(n,) array-like floats: control variable (optimization variable)
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        nsmp int: number of Monte-Carlo samples used for estimation
        cov_matrix (ndarray(m,m), optional): covariance matrix of fluctuations defaults to identity matrix 
    """
    cnter = 0
    if cov_ineqs is None:
        cov_matrix = np.eye(len(x_bn)) 
    else:
        cov_matrix = cov_ineqs
    vec = np.dot(x_An, x) - x_bn
    mean = np.zeros(cov_matrix.shape[-1])
    for i in range(nsmp):
        val = vec + np.random.multivariate_normal(mean, cov_matrix)
        if max(val) <= 0:
            cnter += 1
    return cnter / nsmp

def inner_polyhedral_constraint(x, x_An, x_bn, eta, ):
    """
    Returns the value of constraint from the inner polyhedral approximation
    args:
        x(n,) array-like floats: control variable (optimization variable)
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
    """
    #vec = np.dot(x_An, x) - x_bn
    vec = x_An @ x - x_bn
    cdf_vec = [norm.cdf(v) for v in vec]
    sum_ = np.sum(cdf_vec)
    return sum_ - eta

def inner_polyhedral_constraint_grad(x, x_An, x_bn, eta, eta_var=True):
    """return the gradient of 
        inner polyhedral (union bound) constraint at (x, eta) or x, depends on eta_var
    Args:
        x (ndarray(n,)): point (e.g., generations)
        x_An (ndarray(m,n)): matrix of linear constraints
        x_bn (ndarray(m,)): vector of R.H.S.s
        eta (float): relibaility parameters
        eta_var (bool, optional): if include gradient on eta into the returning value. Defaults to True.

    Returns:
        ndarray(n,) ndarray(n+2,): grad on x or x and eta_1, eta_2
    """
    sum_term_grad_x = norm.pdf(np.dot(x_An, x) - x_bn).reshape(-1, 1) * x_An
    grad_x = sum_term_grad_x.sum(axis=0)
    grad_eta = np.array([0, -1])
    if eta_var:
        full_grad = np.hstack((grad_x, grad_eta))
        return full_grad
    else:
        return grad_x

@njit
def pdf_max(y, mu):
    """
    Return the value of pdf of random variable Y = max_i(X_i), where X_i ~ N(mu_i, 1)
    args:
        y float: value of r.v. Y
        mu array-like floats: expectations of each normally distributed X_i
    """
    out = 0.0
    pdf_ = lambda x: 1. / np.sqrt(2 * np.pi) * np.exp(- 0.5 * x ** 2)
    cdf_ = lambda x: 0.5 * (1. + math.erf(x / math.sqrt(2)))
    for i in range(len(mu)):
        #out += norm.pdf(y - mu[i]) * np.exp(np.sum([np.log(norm.cdf(y - mu[j])) for j in range(len(mu)) if j != i]))
        out += pdf_(y - mu[i]) * np.prod(np.array([(cdf_(y - mu[j])) for j in range(len(mu)) if j != i]))
    return out
def compute_E(foo, density):
    """
    Computes the expectation of foo(x) over distribution with pdf density(x):
    \int foo(x) * density(x) dx from -inf to + inf
    args:
        foo(x) function: r.v. 
        density(x) function: pdf of the distribution 
    """
    res = integrate.quad(lambda y: foo(y) * density(y), -np.inf, np.inf)
    return res[0]

def constr_markov(var, x_An, x_bn, eta):
    """
    Returns the value of Markov approximation constraint
    args:
        var(n + 1,) array-like floats: control variable (optimization variable) `x` and an additional 1d optimization variable `t` appeared from the approximation in order var = [x, t]
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
    """
    x = var[:-1]
    t = var[-1]
    mu = (np.dot(x_An, x) - x_bn)
    density = lambda y: pdf_max(y, mu)
    foo = lambda y: np.max((0., y + t))
    integral = compute_E(foo=foo, density=density)
    return integral - eta * t

def constr_chebyshev(var, x_An, x_bn, eta):
    """
    Returns the value of Chebyshev approximation constraint
    args:
        var(n + 1,) array-like floats: control variable (optimization variable) `x` and an additional 1d optimization variable `t` appeared from the approximation in order var = [x, t]
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
    """
    x = var[:-1]
    t = var[-1]
    mu = (np.dot(x_An, x) - x_bn)
    density = lambda y: pdf_max(y, mu)
    foo = lambda y: np.max((0., y + t)) ** 2
    integral = compute_E(foo=foo, density=density)
    return integral - eta * t **2

def constr_chernoff(var, x_An, x_bn, eta):
    """
    Returns the value of Chernoff approximation constraint
    args:
        var(n + 1,) array-like floats: control variable (optimization variable) `x` and an additional 1d optimization variable `t` appeared from the approximation in order var = [x, t]
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
    """

    x = var[:-1]
    t = var[-1]
    mu = (np.dot(x_An, x) - x_bn)
    density = lambda y: pdf_max(y, mu)
    foo = lambda y: np.exp(y / t)
    integral = compute_E(foo=foo, density=density)
    return np.log(integral) - np.log(eta)
