import sys
import os
from solvers.constr_formulations import *
import nevergrad as ng
from scipy.optimize import NonlinearConstraint, minimize, linprog, OptimizeResult
### NO WARNINGS
import warnings
warnings.filterwarnings("ignore")
### NO WARNINGS
def ground_truth(eta, x_An, x_bn, x0 = [0.0, 0.0], c = None, controllable_idxs = None):
    """
    Returns scipy minimize result of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
    
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
    nl_constraints = [NonlinearConstraint(lambda x: joint_cc_gt(x, x_An, x_bn), lb = [np.log(1. - eta)], ub=[np.inf], keep_feasible=True)]
    if controllable_idxs is None:
        l = -np.ones(len(cost_coeffs)) * np.inf
        u = np.ones(len(cost_coeffs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
        res = minimize(fun=lambda x: objective(cost_coeffs, x), x0=x0, constraints=nl_constraints, method='trust-constr', bounds=bnds)
    else:
        l = np.zeros(len(cost_coeffs))
        u = np.zeros(len(cost_coeffs))
        l[controllable_idxs] = - np.ones(len(controllable_idxs)) * np.inf
        u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
        res = minimize(fun=lambda x: objective(cost_coeffs, x), x0=x0, constraints=nl_constraints, method='trust-constr', bounds=bnds)
    return res

def stage_II_tightening(eta, cost_coeffs, A_n_decorr, b_n_decorr, A_inval_standard, b_inval_standard, x0, keep_feasible=True, r=1e9):
    """
    Returns scipy minimize result of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        cost_coeffs(m,) array-like floats: cost coefficient. already after decorrelating transform
        A_n_decorr(|I|, m) array-like floats: matrix with rows a_i from important subset of planes. already after decorrelation transform
        b_n_decorr(|I|,) array-like-like floats: vector of b_i from important subset of planes. already after decorrelation transform
        A_inval_standard(n-|I|, m) array-like floats: matrix with rows a_i from the complement of important subset of planes. normed on standard deviation of corresponding fluctuation
        b_inval_standard(|I|,) array-like-like floats: vector of b_i from the complement of important subset of planes. normed on standard deviation of corresponding fluctuation
    
    """

    #r = 1e9
    alpha = 10
    l = np.hstack((-np.ones(len(cost_coeffs)) * np.inf, np.array([0.0, 0.0])))
    u = np.ones(len(cost_coeffs)+2) * np.inf
    bnds = [(l[i], u[i]) for i in range(len(l))]
    inner_foo = lambda x_: inner_polyhedral_constraint(x_[:-2], A_inval_standard, b_inval_standard, x_[-1])
    inner_foo_grad = lambda x_: inner_polyhedral_constraint_grad(x_[:-2], A_inval_standard, b_inval_standard, x_[-1])

    main_foo  = lambda x_: -joint_cc_gt(x_[:-2], A_n_decorr, b_n_decorr) + np.log(1.-x_[-2])
    main_foo_grad = lambda x_: joint_cc_grad(x_[:-2], A_n_decorr, b_n_decorr, x_[-2])

    etas_foo  = lambda x_: alpha * (x_[-1] + x_[-2] - eta)**2
    eta1_foo  = lambda x_: -x_[-1]
    eta2_foo  = lambda x_: -x_[-2]
    etas_foo_grad = lambda x_: np.hstack((np.zeros(len(x0)-2), 2 * alpha * (x_[-1] + x_[-2] - eta) * np.array([1., 1.])))
    eta1_foo_grad = lambda x_: np.hstack((np.zeros(len(x0)-2), np.array([0., -1.])))
    eta2_foo_grad = lambda x_: np.hstack((np.zeros(len(x0)-2), np.array([-1., 0.])))
    
    ###SCIPY                                  
    res = minimize(fun=lambda x: objective_barrier_multiple(cost_coeffs, x,\
                [inner_foo, main_foo, etas_foo, eta1_foo, eta2_foo], r),\
                     jac=lambda x: objective_barrier_multiple_grad(cost_coeffs, x, [inner_foo, main_foo, etas_foo, eta1_foo, eta2_foo], [inner_foo_grad, main_foo_grad, etas_foo_grad, eta1_foo_grad, eta2_foo_grad], r),\
                   x0=x0, bounds=bnds, method='COBYLA') #CG ok but slow, BFGS not ok
    print("Optimiziation Succeeded:", res.success)
    ###GRAD
    # objective = lambda x: objective_barrier_multiple(cost_coeffs, x,\
    #              [inner_foo, main_foo, etas_foo], r)
    # grad = lambda x: objective_barrier_multiple_grad(cost_coeffs, x, [inner_foo, main_foo, etas_foo], [inner_foo_grad, main_foo_grad, etas_foo_grad], r)
    # xk = x0
    # for iter_num in range(2000):
    #     xk = xk - 1e-8 * grad(xk)
    # res = OptimizeResult()
    # res["x"] = xk
    # res["fun"] = objective(xk)
    return res

def outer_polyhedral(eta, x_An, x_bn, c = None, controllable_idxs = None):
    """
    Returns scipy minimize result of the outer approximation of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
    
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
    
    if controllable_idxs is None:
        l = np.zeros(x_An.shape[1])
        u = np.ones(x_An.shape[1]) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    else:
        l = np.zeros(len(cost_coeffs))
        u = np.zeros(len(cost_coeffs))
        u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    res = linprog(c = cost_coeffs, A_ub = x_An, b_ub = x_bn - norm.ppf(1-eta), bounds=bnds)
    return res

def inner_polyhedral(eta, x_An, x_bn, r=1000000., x0=[-0.1,-0.1], c = None, controllable_idxs = None):
    """
    Returns scipy minimize result of the inner approximation of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        r float: penalization parameter
        x0 (n,) array-like floats: initial guess
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
    if controllable_idxs is None:
        l = -np.ones(x_An.shape[1]) * np.inf
        u = np.ones(x_An.shape[1]) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    else:
        l = np.zeros(len(cost_coeffs))
        u = np.zeros(len(cost_coeffs))
        l[controllable_idxs] = - np.ones(len(controllable_idxs)) * np.inf
        u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    #res = minimize(fun=lambda x: objective_barrier(cost_coeffs, x, lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta), r), x0=x0, bounds=bnds, method='Nelder-Mead')
    #res = minimize(fun=lambda x: objective_barrier(cost_coeffs, x, , x0=x0, bounds=bnds, method='Nelder-Mead')
    ###SCIPY
    res = minimize(fun=lambda x: objective_barrier_multiple(cost_coeffs, x,\
                [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)], r),\
                     jac=lambda x: objective_barrier_multiple_grad(cost_coeffs, x, [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)],\
                          [lambda x_: inner_polyhedral_constraint_grad(x_, x_An, x_bn, eta, eta_var=False)], r=r, eta_var=False),\
                   x0=x0, bounds=bnds, method='COBYLA')
    print("Optimiziation Succeeded:", res.success)
    ####GRAD
    # objective = lambda x: objective_barrier_multiple(cost_coeffs, x,\
    #              [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)], r)
    # grad = lambda x: objective_barrier_multiple_grad(cost_coeffs, x, [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)],\
    #                        [lambda x_: inner_polyhedral_constraint_grad(x_, x_An, x_bn, eta, eta_var=False)], r=r, eta_var=False)
    # xk = x0
    # for iter_num in range(2000):
    #     xk = xk - 1e-6 * grad(xk)
    # res = OptimizeResult()
    # res["x"] = xk
    # res["fun"] = objective(xk)
    ####NG
    # def objective(x):
    #     return objective_barrier_multiple(cost_coeffs, x, [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)], r)
    # parametrization = ng.p.Instrumentation(x=ng.p.Array(shape=(len(x0),)))
    # optimizer = ng.optimizers.NGOpt(parametrization=parametrization,budget=100)
    # recommendation = optimizer.minimize(objective)
    # res = OptimizeResult()
    # res["x"] = recommendation[1]["x"].value
    # res["fun"] = cost_coeffs.dot(res["x"])
    return res

def markov(eta, x_An, x_bn, r=1000000., x0=[0.1, 0.1, 0.1], c = None, controllable_idxs = None):
    """
    Returns scipy minimize result of the Markov approximation of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        r float: penalization parameter
        x0 (n+1,) array-like floats: initial guess
            *see 'constr_formulations.py -> constr_markov(var, x_An, x_bn)' for details
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
    if controllable_idxs is None:
        l = np.zeros(x_An.shape[1] + 1)
        u = np.ones(x_An.shape[1] + 1) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    else:
        l = np.zeros(len(cost_coeffs) + 1)
        u = np.concatenate((np.zeros(len(cost_coeffs)), [np.inf]), axis=-1)
        u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    res = minimize(fun=lambda x: objective_barrier(cost_coeffs, x, lambda x_: constr_markov(x_, x_An, x_bn, eta), r), x0=x0, bounds=bnds, method='Nelder-Mead')
    return res

def chebyshev(eta, x_An, x_bn, r=1000000., x0=[0.1, 0.1, 0.1], c = None, controllable_idxs = None):
    """
    Returns scipy minimize result of the Chebyshev approximation of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        r float: penalization parameter
        x0 (n+1,) array-like floats: initial guess
            *see 'constr_formulations.py -> constr_markov(var, x_An, x_bn)' for details
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
    if controllable_idxs is None:
        l = np.zeros(x_An.shape[1] + 1)
        u = np.ones(x_An.shape[1] + 1) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    else:
        l = np.zeros(len(cost_coeffs) + 1)
        u = np.concatenate((np.zeros(len(cost_coeffs)), [np.inf]), axis=-1)
        u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    res = minimize(fun=lambda x: objective_barrier(cost_coeffs, x, lambda x_: constr_chebyshev(x_, x_An, x_bn, eta), r), x0=x0, bounds=bnds, method='Nelder-Mead')
    return res

def chernoff(eta, x_An, x_bn, r=1000000., x0=[0.1, 0.1, 5.1], c = None, controllable_idxs = None):
    """
    Returns scipy minimize result of the Chebyshev approximation of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''

    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        r float: penalization parameter
        x0 (n+1,) array-like floats: initial guess
            *see 'constr_formulations.py -> constr_markov(var, x_An, x_bn)' for details
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
    if controllable_idxs is None:
        l = np.zeros(x_An.shape[1] + 1)
        u = np.ones(x_An.shape[1] + 1) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    else:
        l = np.zeros(len(cost_coeffs) + 1)
        u = np.concatenate((np.zeros(len(cost_coeffs)), [np.inf]), axis=-1)
        u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
        bnds = [(l[i], u[i]) for i in range(len(l))]
    res = minimize(fun=lambda x: objective_barrier(cost_coeffs, x, lambda x_: constr_chernoff(x_, x_An, x_bn, eta), r), x0=x0, bounds=bnds, method='Nelder-Mead')
    return res

def scenario_approx(x_An, x_bn, nsmp, c = None):
    """
    Returns scipy minimize result of the scenario approximation of the original chance constrained problem based on `nsmp` samples: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''
    
    args:
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        nsmp int: number of samples for the approximation
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c
#     if controllable_idxs is None:
#         l = np.zeros(x_An.shape[1] + 1)
#         u = np.ones(x_An.shape[1] + 1) * np.inf
#         bnds = [(l[i], u[i]) for i in range(len(l))]
#     else:
#         l = np.zeros(len(cost_coeffs) + 1)
#         u = np.concatenate((np.zeros(len(cost_coeffs)), [np.inf]), axis=-1)
#         u[controllable_idxs] = np.ones(len(controllable_idxs)) * np.inf
#         bnds = [(l[i], u[i]) for i in range(len(l))]
    Xi = [np.random.normal(0, 1, x_An.shape[0]) for i in range(nsmp)]
    x_An_scenario = np.concatenate([x_An for i in range(nsmp)])
    x_bn_scenario = np.concatenate([x_bn + Xi[i] for i in range(nsmp)])
    res = linprog(c = cost_coeffs, A_ub = x_An_scenario, b_ub = x_bn_scenario)
    return res