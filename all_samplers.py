import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss, wasserstein_distance
from dataclasses import dataclass  

@dataclass
class SamplerConfig:
    n_samples = int(1e5)
    burn_in = int(1e3)
    
    a = 1.0
    b = 3.0
    beta = 2.0
    lam = 2.7
    
    def get_params(self):
        return self.a, self.b, self.beta, self.lam, self.n_samples, self.burn_in
    
    def posterior(self, xx):
        G = 0.5*(self.a*xx-self.b)**2
        R = self.lam * np.abs(xx)
        
        rho = np.exp(-self.beta*(G+R))
        Z = np.trapezoid(rho, xx)
        return rho/Z
    
    def make_grad_G(self):
        def grad_G(x):
            return self.a*(self.a*x - self.b)
        return grad_G
    
    def true_mean(self):
        xx = np.linspace(-10, 10, 1000)
        yy = self.posterior(xx)
        return np.trapezoid(xx * yy, xx)
    
    def plot_sample(self, sample, name, ax):
        if ax is None:
            ax = plt.gca()
        x = np.linspace(np.min(sample) - 1, np.max(sample) + 1, 1000)
        ax.plot(x, self.posterior(x), color='red', label='Target distribution')        
        ax.hist(sample, bins=100, density=True, alpha=0.5, color='steelblue', edgecolor='black', label=f'{name} samples')
        ax.set_xlim(np.min(sample) - 1, np.max(sample) + 1)
        
        for name, metric in zip(['Mean', 'Variance', 'MSE', 'Wasserstein'], [np.mean, np.var, self.mse_mean, self.wassterstein]):
            ax.plot(0, 0, color='none', label=f'{name}: {metric(sample):.2f}')
        
        ax.legend()
        
    def plot_all_samples(self, samples, names):
        fig, axes = plt.subplots(1, len(samples), figsize=(5*len(samples), 4))
        for sample, name, ax in zip(samples, names, axes):
            self.plot_sample(sample, name, ax)
        return fig, axes
    
    def mse_mean(self, sample):
        return (np.mean(sample) - self.true_mean())**2
    
    def wassterstein(self, sample):
        xx = np.linspace(np.min(sample) - 1, np.max(sample) + 1, len(sample))
        yy = self.posterior(xx)
        return wasserstein_distance(xx, sample, u_weights=yy)
        



def gibbs_sampler(config: SamplerConfig, SEED=0, tol = 1e-10):
    a, b, beta, lam, n_samples, burn_in = config.get_params()
    np.random.seed(SEED)
    tol = tol
    
    samples = []
    
    x = 0.0
    eta = 1.0
    
    for i in range(n_samples + burn_in):
        C = (1.0 / eta) + beta * a**2 
        sigma = 1.0/C
        mu = beta * sigma * a*b
        
        x = np.random.normal(mu, sigma**0.5)
        
        nu = beta * lam / (np.abs(x) + tol)
        sc = (beta * lam)**2
            
        z = invgauss.rvs(mu=nu/sc, scale=sc)
        eta = 1.0 / z
        
        if i >= burn_in:
            samples.append(x)
    
    return np.array(samples)

def make_pi_gamma(gamma, config: SamplerConfig):
    def g_gamma(x):
        xx = np.linspace(-10, 10, 1000)
        R = config.lam * np.abs(xx)
        dist = (xx - x)**2 / (2*gamma)
        return np.min(R + dist)
    def pi_gamma(x):
        G = 0.5*(config.a*x-config.b)**2
        pi = np.exp(-config.beta * (g_gamma(x)+G))
        return pi
    return np.vectorize(pi_gamma)

def prox_l1(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def myula_sampler(gamma, h, config: SamplerConfig, SEED=0):
    np.random.seed(SEED)
    a, b, beta, lam, n_samples, burn_in = config.get_params()
    G_prime = config.make_grad_G()

    x = 0.0
    samples = []

    for i in range(n_samples+burn_in):
        grad_G = G_prime(x)

        prox = prox_l1(x, gamma * lam)

        grad_R_gamma = (x - prox) / gamma

        noise = np.sqrt(2 * h / beta) * np.random.randn()

        x = x - h * (grad_G + grad_R_gamma) + noise

        if i >= burn_in:
            samples.append(x)

    return np.array(samples)

def hadamard_sampler(h, config: SamplerConfig, SEED=0):
    np.random.seed(SEED)
    a, b, beta, lam, n_samples, burn_in = config.get_params()
    G_prime = config.make_grad_G()

    samples = []

    # initial values
    u = 1.0
    v = 1.0

    for i in range(n_samples + burn_in):
        
        dW1 = np.sqrt(h) * np.random.randn()
        dW2 = np.sqrt(h) * np.random.randn()
    
        grad_G = G_prime(u*v)
    
        num = v - u*grad_G*h + np.sqrt(2/beta)*dW2
        v = num/(1 + lam*h)
    
        C = u - v*grad_G*h + np.sqrt(2/beta)*dW1
    
        a = (1 + lam*h)
        b = -C
        c = -h/beta
    
        disc = b**2 - 4*a*c
        u = (-b + np.sqrt(disc))/(2*a)
        
        if i >= burn_in:
            samples.append(u * v)
    return np.array(samples)




def main_gibbs():
    config = SamplerConfig()
    config.plot_sample(gibbs_sampler(config), 'Gibbs')
    print('Done')

    
# main_gibbs()

def main_myula():
    gamma, h = 0.5, 0.05
    config = SamplerConfig()
    config.plot_sample(myula_sampler(gamma, h, config), 'MYULA')
    print('Done')
    
# main_myula()

def main_hadamard():
    h = 0.05
    config = SamplerConfig()
    config.plot_sample(hadamard_sampler(h, config), 'Hadamard')
    print('Done')
    
# main_hadamard()

def main_all(gamma=0.01, h=0.05):

    config = SamplerConfig()
    sample_size = config.n_samples
    print('Sampling...')
    sample_gib = gibbs_sampler(config)
    print('Done Gibbs sampling')
    sample_myula = myula_sampler(gamma, h, config)
    print('Done MYULA sampling')
    sample_hadamard = hadamard_sampler(h, config)
    print('Done Hadamard sampling')

    names = ['Gibbs', 'MYULA', 'Hadamard']
    all_samples = [sample_gib, sample_myula, sample_hadamard]
    mn, mx = np.min(all_samples), np.max(all_samples)

    xx = np.linspace(mn - 1, mx + 1, sample_size)
    # yy = config.posterior(xx)

    # mse = lambda sample: np.mean((yy - sample) ** 2)
    # wasserstein = lambda sample: wasserstein_distance(yy, sample)
    # for metric in [np.mean, np.var, mse, wasserstein]:
    #     print(f'{metric.__name__} for the true distribution: {metric(yy):.4f}')
    #     for name, sample in zip(names, all_samples):
    #         print(f'For {name}: {metric(sample):.4f}')

    n_samples, burn_in = config.n_samples, config.burn_in
    pi_gamma = make_pi_gamma(gamma, config)

    fig, axs = config.plot_all_samples(all_samples, names)
    fig.suptitle(f'Comparison of Samplers. n_samples={n_samples:.0e}, burn_in={burn_in:.0e}', fontsize=16)
    axs[1].plot(xx, pi_gamma(xx), color='green', label=r'$\pi_\gamma$')
    axs[1].legend()
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main_all()