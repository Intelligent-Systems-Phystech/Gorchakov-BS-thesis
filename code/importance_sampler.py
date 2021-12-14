import math
import numpy as np
import scipy.linalg as la

from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import multinomial
import itertools

from tqdm.notebook import tqdm

class ImportanceEstimator:
    def __init__(self, small_prob_th=0.001, n_samples=1000, eta_vm=0.1, eps=0.001):
        ### Cut small probabilities threshold
        ### discard all probabilities than thrs* prb(closest hyperplane)
        ### Crucially affects time performance
        #thrs = 0.001
        #self.n_samples = 1000 
        ### Step-sizes for KL and Var minimization 
        #eta_vm = 0.1 
        #eta_kl = 0.1 
        ### Rounding threshold in optimization: 
        ### if a (normalized on the simplex) hpl probability becomes lower then 0.001
        ### we increase it to this level
        ### Crucially affects numerical stability
        #eps = 0.001
        self.small_prob_th = small_prob_th
        self.n_samples = n_samples
        self.eta = eta_vm
        self.eps = eps
        ### Compute probabilities:
        ### prb: probability of each hpl failure 
        ### p_up, p_dwn: upper and lower bounds
        ###
    def estimate(self, A_n, b_n):
        prb = norm.cdf(-b_n)
        p_up = np.sum(prb)
        p_dwn = np.max(prb)
        n = A_n.shape[1]
        print("the union bound (upper):", p_up)
        print("the max bound (lower):", p_dwn)
        z = norm.rvs(size=[self.n_samples, n])
        u = uniform.rvs(size=[self.n_samples])
        ### Keep only valuable probabilities: 
        ### - use the union bound for all the rest
        ### - keep only the prbs higher than the thrs* p_dwn
        prbh_id = (prb > self.small_prob_th * p_dwn)
        prb_rmd = np.sum(prb[~(prb > self.small_prob_th * p_dwn)])
        print("Remainder probability (omitted):", prb_rmd)
        ############ Preliminary steps for Sampling and Importance Sampling ############

        ### normalize all active probabilities to one 
        ### as we only play a hyperplane out of them
        ###
        ### NB: crucial steps in performance optimization
        ###

        x_id = np.where(prbh_id == True)[0]

        ### local normalized versions of A and b, 
        ### reduced in size: number of rows now is equal to a number of constraints
        ### that have a high probability of violation
        ###

        ### we do not care about the full matrix A and vector b
        ### only about important parts of them
        A_n = np.array(A_n)
        A_n = A_n[x_id]
        b_n = b_n[x_id]

        ### setup the initial values
        x_alph = prb[prbh_id] / np.sum(prb[prbh_id])
        md_var = 0 
        md_exp = 0
        grad = np.zeros(len(b_n)) #gradient on each iteration
        _hpl = np.zeros(self.n_samples) # hpls choosen by the method 
        ### intentionally use a copy instead of a reference
        ### alph is a vector of weigths to be updated in algorithm
        ###

        alph = x_alph[:]

        ### history of probability estimate and std
        md_exp_history = []
        md_std_history = []

        ### grad normalization by prbs[i] factor is introduced to make computations numerically stable
        ###

        prbs = prb[prbh_id]

        for i in tqdm(range(0,self.n_samples)):
            
            ### sample x according to current alph
            hpl_id = np.where(multinomial.rvs(n=1,p = alph, size=1, random_state=None)[0] == 1)[0]
            _hpl[i] = hpl_id        
            
            ### generate a sample following to the ALOE procedure
            y = norm.ppf(u[i]*norm.cdf(-b_n[hpl_id]))
            
            x_smp = - A_n[hpl_id] * y - z[i] + np.outer(A_n[hpl_id], A_n[hpl_id]) @ z[i]
            
            ### the RHS' to be compared with b_n
            x_smp = A_n @ x_smp.T

            ### results of constraints violations for each generated object
            cns_vlt = (b_n <= x_smp.T[:])[0]
            
            ### weight vector defined by the multiplicity of constraint violation for each sample
            wgt = 1./np.sum(np.multiply(cns_vlt, np.multiply(alph, 1. / x_alph))) 
            
            
            ### compute gradient of the variance, see the paper (our + OMC) for details
            grad = [-p_up * p_up * wgt * wgt * norm.pdf(x_smp[k])[0] * cns_vlt[k] / prbs[k] for k in range(len(x_smp))]
            grad = np.array(grad)

            
            ### The gradient is high -- signal about emergency as it can zero out all weights
            if (la.norm(self.eta * grad) > 1e4):
                print("\n##############    Extremely high gradient      ############\n")
                print("Iteration: ", i, "\nGradient:", grad)
            
            ### make a ``simplex MD'' update
            alph = [math.exp(-self.eta * grad[k]) * alph[k] for k in range(0, len(x_smp))]
            

            ### enter if some coordinates are too small and may cause numerical instability
            ### increase the corresponding weigths
            if (np.min(alph) < self.eps):
                print("###########  some coordinates are small  #################")
                alph = [alph[k] + self.eps for k in range(0, len(b_n))]
                
            ### make a projection to the unit simplex
            alph = alph/np.sum(alph) 
            
            ### adjust contribution to the errors
            md_exp = md_exp + wgt
            md_exp_history.append(p_up * md_exp / (i + 1) + prb_rmd)
            md_var = md_var + p_up * np.dot(grad.T, grad)
            md_std_history.append(p_up * math.sqrt(md_var) / (i + 1))
            
            
        print("Optimal weigths of MD-Var minimization: ", alph)
        print("Optimal weigths of ALOE", x_alph)

        ### normalize errors, compute standard deviation
        md_exp = p_up * md_exp / self.n_samples + prb_rmd 
        md_std = p_up * math.sqrt(md_var) / self.n_samples

        print("MD-Var (exp, std)", (md_exp, md_std))
        return md_exp


class ALOEEstimator:
    def __init__(self, small_prob_th=0.001, n_samples=1000):
        self.n_samples = n_samples
        self.small_prob_th = small_prob_th

    def estimate(self, A_n, b_n):
        ############# ALOE ##################
        ### 
        ### Exactly follows to the Owen/Maximov/Chertkov paper, EJOS'19
        ###
        ### sample z ~ N(0, I_n)
        ### sample u ~ U(0,1)
        ### compute y = F^{-1}(u F(-b_i))
        ### compute x = - (a_i * y + (I - a_i.T * a_i) z)
        ###
        ### Ouput: union bound divided by the expected failure multiplicity
        ###


        ### Initialize samplers
        ###
        ### sample z ~ N(0, I_n) and u ~ U(0,1)
        ### 
        prb = norm.cdf(-b_n)
        p_up = np.sum(prb)
        p_dwn = np.max(prb)
        n = A_n.shape[1]
        z = norm.rvs(size=[self.n_samples, n]) 
        u = uniform.rvs(size=[self.n_samples])
        print("the union bound (upper):", p_up)
        print("the max bound (lower):", p_dwn)
        ### Keep only valuable probabilities: 
        ### - use the union bound for all the rest
        ### - keep only the prbs higher than the thrs* p_dwn
        prbh_id = (prb > self.small_prob_th * p_dwn)
        prb_rmd = np.sum(prb[~(prb > self.small_prob_th * p_dwn)])
        print("Remainder probability (omitted):", prb_rmd)

        ### x_alph is a vector of ALOE probabilities 
        ### normalized by a unit simplex
        ###

        x_alph = prb[prbh_id] / np.sum(prb[prbh_id])
        print("ALOE prbs for major hpls: ", x_alph)

        ### _hpl: how many smpls beyond each of the hpls
        ###

        _hpl = multinomial.rvs(n = self.n_samples, p = x_alph)

        ### print("# samples per hpl", _hpl)

        ### Get cummulative sums, which are easier to work with
        _hpl = list(itertools.accumulate(_hpl))
        _hpl = np.array(_hpl)

        ### normalize all active probabilities to one 
        ### as we only play a hyperplane out of them
        ###
        ### NB: crucial steps in performance optimization
        ###

        x_id = np.where(prbh_id == True)[0]

        ### we do not care about the full matrix A and vector b
        ### only about important parts of them
        A_n = np.array(A_n)
        A_n = A_n[x_id]
        b_n = b_n[x_id]

        ### local normalized versions of A and b, 
        ### reduced in size: number of rows now is equal to a number of constraints
        ### that have a high probability of violation
        ###

        ### Generate samples
        ### x_aloe -- samples generated by ALOE
        ###
        ### TODO: seems optimizable, but I am not sure about memory mgmnt in python
        x_aloe = np.zeros([self.n_samples, n]) 

        # index of the active hyperplane
        hpl_id = 0

        ### get samples x_aloe according to the algorithm
        #for i in tqdm(range(0,nsmp)):
        for i in range(0,self.n_samples):
            ### get index of a hyperplane to sample beyond
            hpl_id = (hpl_id, hpl_id+1)[i >= _hpl[hpl_id]]
            y = norm.ppf(u[i] * norm.cdf(-b_n[hpl_id]))
            x_aloe[i] = - A_n[hpl_id] * y - z[i] + np.outer(A_n[hpl_id], A_n[hpl_id]) @ z[i]

        ### test how many constraints are violated
        smp = A_n @ x_aloe.T

        ### compute expectation and std final and history
        aloe_exp = p_up * np.sum(1. / np.sum(b_n <= smp.T[:],axis=1)) / self.n_samples + prb_rmd 
        aloe_std = p_up * math.sqrt(2 * len(_hpl)) / math.sqrt(self.n_samples) # indeed len(_hpl) instead of 2*m in the Thrm
        aloe_exp_history = [p_up * np.sum(1. / np.sum(b_n <= (A_n @ x_aloe[:i, :].T).T, axis=1)) / (i + 1) + prb_rmd for i in range(0,self.n_samples)]
        aloe_std_history = [p_up * math.sqrt(2*len(_hpl))/math.sqrt(i + 1) for i in range(0, self.n_samples)]

        print("ALOE (exp, std)", (aloe_exp, aloe_std))

        return aloe_exp_history[-1]


