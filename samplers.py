import numpy as np
from numpy.linalg import inv
from scipy.stats import invgamma, norm,invgauss



def one_step_langevin(x,p, grad, tau,beta=1):
    """
    This function implements one step of proximal langevin. To sample from exp(-beta*(G(x)))
        where G is smooth and R is nonsmooth.
        Returns x' = x - tau*grad_G(x) + sqrt(2*tau/beta)*N(0,I_p)

    :param x: initial vector size (p,)
    :param p: int, length of x
    :param tau: float, stepsize
    :param grad: gradient of G. This is a function mapping 
        numpy.ndarray of size (p,) to numpy.ndarray of size (p,)
    :param beta: float, inverse temperature, beta=1 by default.
    :return y: numpy.ndarray of size (p,)
        next iteration of proximal langevin
    """
    y = x - tau*grad(x) + np.sqrt(2*tau/beta)*np.random.randn(p,)
    return y



def one_step_MALA(x, p, fval, grad, tau, beta=1):
    """
    This function implements one step of metropolis hasting proximal langevin. To sample from exp(-beta*G(x))
        where G is smooth and R is nonsmooth.
        Returns x' = x - tau*(grad_G(x) + (x - prox_R(x,gamma))/gamma) + sqrt(2*tau/beta)*N(0,I_p)

    :param x: initial vector size (p,)
    :param p: int, length of x
    :param tau: float, stepsize
    :param fval: G. This is a function mapping 
        numpy.ndarray of size (p,) to float
    :param grad: gradient of smooth term. This is a function mapping 
        numpy.ndarray of size (p,) to numpy.ndarray of size (p,)
    :param beta: float, inverse temperature, beta=1 by default.
    :return y: numpy.ndarray of size (p,)
        next iteration of proximal langevin
    """
    
    #propose new point
    Y = lambda x: x - tau*grad(x)
    x_ = Y(x) + np.sqrt(2*tau/beta)*np.random.randn(p,)
    
    log_pi = lambda x:  -beta*fval(x) 

    #probability of transitining from x to x_
    def log_q(x_, x):
        return  -beta/4/tau * np.linalg.norm(x_-Y(x))**2

    # Compute Metropolis-Hastings acceptance probability
    log_acceptance_ratio = log_pi(x_) - log_pi(x)
    log_acceptance_ratio += log_q(x,x_) - log_q(x_,x)
    if np.log(np.random.uniform()) <= log_acceptance_ratio:
        return x_ #accept proposal
    else: 
        return x


def one_step_hadamard(x, p, grad, tau, lam,beta=1):
    """
    This function implements one step of Hadamard langevin. To sample from exp(-beta*(G(x)+lam*|x|_1))
        where G is smooth.

    :param x: initial vector size (2*p,) representing (u,v)
    :param p: int, .5 * length of x
    :param tau: float, stepsize
    :param grad: gradient of smooth term. This is a function mapping 
        numpy.ndarray of size (p,) to numpy.ndarray of size (p,)
    :param lam: float, regularization parameter for l1 term
    :param beta: float, inverse temperature, beta=1 by default.
    :return y: numpy.ndarray of size (2*p,)
        next iteration of hadamard langevin
    """
    

    u = x[:p]
    v = x[p:]

    g = grad(u*v)
    Grad = np.concatenate((v*g, u*g))
    z = x - tau*Grad + np.random.randn(2*p,)*np.sqrt(2*tau/beta)
    z[:p] = (z[:p]+np.sqrt(z[:p]**2 + 4*tau/beta*(1+tau*lam)))/2
    x_ = z/(1+tau*lam)
    return x_
    

def one_step_MALA_hadamard(x, p, fval, grad, tau, lam, beta=1):
    """
    This function implements one step of Hadamard langevin. To sample from exp(-beta*(G(x)+lam*|x|_1))
        where G is smooth.

    :param x: initial vector size (2*p,) representing (u,v)
    :param p: int, .5 * length of x
    :param tau: float, stepsize
    :param fval: functional value of smooth part to negative log density
    :param grad: gradient of smooth term in negative log density. This is a function mapping 
        numpy.ndarray of size (p,) to numpy.ndarray of size (p,)
    :param lam: float, regularization parameter for l1 term
    :param beta: float, inverse temperature, beta=1 by default.
    :return y: numpy.ndarray of size (2*p,)
        next iteration of hadamard langevin
    """
    
    ru = lambda x: x[:p]
    rv = lambda x: x[p:]
    prod = lambda x,g: np.concatenate((rv(x)*g ,ru(x)*g))
    Grd = lambda x: prod( x, grad( ru(x)*rv(x) ) )
    # S1 = lambda x: (x+np.sqrt(x**2 + 4/beta*tau*(1+tau*lam)))/2

    #propose new point
    z = x - tau*Grd(x) + np.random.randn(2*p,)*np.sqrt(2*tau/beta)
    z[:p] = (z[:p]+np.sqrt(z[:p]**2 + 4/beta*tau*(1+tau*lam)))/2
    # z[:p] = S1(z[:p])
    x_ = z/(1+tau*lam)


    
    log_pi = lambda x:  (-0.5*lam*np.sum(x**2) - fval(ru(x)*rv(x) ))*beta  + np.sum(np.log(ru(x)))

    #probability of transitining from x to x_
    def log_q(x_, x):
      u_ = ru(x_)
      u = ru(x)
      g = (1+tau*lam)*x_ - x + tau*Grd(x)
      g[:p] = g[:p] - tau/u_/beta
      q = -g**2 * beta/(4*tau)
      q[:p] = q[:p] + np.log(1+tau*lam + tau/u_**2/beta)
      return np.sum(q)

    # Compute Metropolis-Hastings acceptance probability
    log_acceptance_ratio = log_pi(x_) - log_pi(x)
    log_acceptance_ratio += log_q(x,x_) - log_q(x_,x)
    if np.log(np.random.uniform()) <= log_acceptance_ratio:
        return x_ #accept proposal
    else: 
        return x



# Gibbs Sampler for Bayesian Lasso with sigma^2 = 1
def gibbs_sampler(A,y, lam,init, n, burn_in=1000, beta=1):
    """
    This function implements one step of Gibbs sampler for the Bayesian lasso. To sample from exp(-beta*(|A@x-y|^2/2+lam*|x|_1))

    :param A: numpy.ndarray of size (m,p), data matrix
    :param y: numpy.ndarray of size (m,), data vector
    :param lam: float, regularization parameter for l1 term
    :param init: int, initial vector of size (p,) 
    :param n: int,  number of samples to return
    :param burn_in: int, number of iterations to run before recording the samples
    :param beta: float, inverse temperature, beta=1 by default.
    :return samples: numpy.ndarray of size (n,p), generated samples
    """
    
    p = len(init)
    eta = init

    def one_step():
        # Sample x | y, X, eta
        V_x = inv(beta*A.T@A + np.diag(1/eta))
        m_x = beta* V_x @ A.T @ y
        x = np.random.multivariate_normal(m_x, V_x)
    
        # Sample eta_j | x_j
        for j in range(p):
            eta[j]= 1/invgauss.rvs(mu=abs(1./(beta*lam*np.abs(x[j]))), scale=(lam*beta)**2)
        return x
        
    #burn in 
    for i in range(burn_in):
        one_step()

    #record n samples
    samples = np.zeros((n,p))
    for i in range(n):
        x = one_step()
        samples[i,:] = x.reshape(-1)
        
    return samples
    


def generate_samples_x(Iterate,init, n, burn_in=1000):
    """
    This function generates n samples using some sampling mechanisim given by Iterate.

    :param Iterate: function that takes input x_t of size (p,) and outputs x_{t+1} of size (p,)
    :param init: int, initial vector of size (p,) 
    :param n: int,  number of samples to return
    :param burn_in: int, number of iterations to run before recording the samples
    :return samples: numpy.ndarray of size (n,p)
    """
    x = init
    p = len(init)

    #burn in 
    for i in range(burn_in):
        x = Iterate(x)
    # p = len(x)

    #record n samples
    samples = np.zeros((n,p))
    for i in range(n):
        x = Iterate(x)
        samples[i,:] = x.reshape(-1)
        
    return samples
    

def generate_samples_stride(Iterate,init, n, stride=1, burn_in=1000):
    """
    This function generates n samples using some sampling mechanisim given by Iterate.

    :param Iterate: function that takes input x_t of size (p,) and outputs x_{t+1} of size (p,)
    :param init: int, initial vector of size (p,) 
    :param n: int,  number of samples to return
    :param stride: int, number of samples to skip before recording
    :param burn_in: int, number of iterations to run before recording the samples
    :return samples: numpy.ndarray of size (n,p)
    """
    x = init

    #burn in 
    for i in range(burn_in):
        x = Iterate(x)
    p = len(x)

    #record n samples
    samples = np.zeros((n,p))
    
    k=0
    for i in range(n*stride):
        x = Iterate(x)
        if np.mod(i,stride)==0:
            samples[k,:] = x.reshape(-1)
            k+=1
        
    return samples