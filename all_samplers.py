import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss, wasserstein_distance
from dataclasses import dataclass
import arviz as az

@dataclass
class SamplerConfig:
    n_samples = int(1e6)
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
        xx = np.linspace(-20, 20, 50000)
        yy = self.posterior(xx)
        return np.trapezoid(xx * yy, xx)

    def true_var(self):
        xx = np.linspace(-20, 20, 50000)
        yy = self.posterior(xx)
        
        mean = np.trapezoid(xx * yy, xx)
        second_moment = np.trapezoid(xx**2 * yy, xx)
        
        return second_moment - mean**2
    
    def plot_sample(self, sample, name, ax=None):
        if ax is None:
            ax = plt.gca()
        x = np.linspace(np.min(sample) - 1, np.max(sample) + 1, 1000)
        ax.plot(x, self.posterior(x), color='red', label='Target distribution')        
        ax.hist(sample, bins=100, density=True, alpha=0.5, color='steelblue', edgecolor='black', label=f'{name} samples')
        ax.set_xlim(np.min(sample) - 1, np.max(sample) + 1)
        
        for name, metric in zip(['Mean', 'Variance', 'MSE', 'Wasserstein', 'ESS'], [np.mean, np.var, self.mse_first_moment, self.wassterstein, self.effective_sample_size]):
            ax.plot(0, 0, color='none', label=f'{name}: {metric(sample):.4f}')
        
        ax.legend()
        
    def plot_all_samples(self, samples, names):
        fig, axes = plt.subplots(1, len(samples), figsize=(5*len(samples), 4))
        for sample, name, ax in zip(samples, names, axes):
            self.plot_sample(sample, name, ax)
        return fig, axes
    
    def mse_first_moment(self, sample):
        return (np.mean(sample) - self.true_mean())**2
    
    def mse_second_moment(self, sample):
        # E[X^2] = E[X]^2 + Var(X)
        true_second_moment = self.true_mean()**2 + self.true_var()
        sample_second_moments = np.mean(sample**2)
        return np.mean((sample_second_moments - true_second_moment)**2)
    
    def wassterstein(self, sample):
        xx = np.linspace(np.min(sample) - 1, np.max(sample) + 1, len(sample))
        yy = self.posterior(xx)
        return wasserstein_distance(xx, sample, u_weights=yy)
    
    def effective_sample_size(self, sample):
        return az.ess(sample)


def gibbs_sampler(config: SamplerConfig, SEED=0, tol=1e-10):
    a, b, beta, lam, n_samples, burn_in = config.get_params()
    np.random.seed(SEED)
    
    samples = []
    
    x = 0.0
    eta = 1.0
    
    for _ in range(n_samples + burn_in):
        C = (1.0 / eta) + beta * a**2 
        sigma = 1.0/C
        mu = beta * sigma * a*b
        
        x = np.random.normal(mu, sigma**0.5)
        
        nu = beta * lam / (np.abs(x) + tol)
        sc = (beta * lam)**2
            
        z = invgauss.rvs(mu=nu/sc, scale=sc)
        eta = 1.0 / z
        
        samples.append(x)
    
    return np.array(samples[burn_in:])

def make_pi_gamma(gamma, config: SamplerConfig):
    def H_gamma(x):
        G = 0.5 * (config.a * x - config.b) ** 2          # smooth quadratic term
        z = prox_l1(x, gamma * config.lam)                               # prox step (vectorised)
        envelope = config.lam * np.abs(z) + (z - x) ** 2 / (2 * gamma)  # Moreau envelope
        return G + envelope

    def pi_gamma(x):
        return np.exp(-config.beta * H_gamma(x))           # fully vectorised now

    xx = np.linspace(-20, 20, 50000)
    Z = np.trapezoid(pi_gamma(xx), xx)
    return lambda x: pi_gamma(x) / Z



def true_mean_gamma(gamma, config):
    xx = np.linspace(-20, 20, 50000)
    pi_gamma = make_pi_gamma(gamma, config)
    yy = pi_gamma(xx)
    return np.trapezoid(xx * yy, xx)

def true_var_gamma(gamma, config):
    xx = np.linspace(-20, 20, 50000)
    pi_gamma = make_pi_gamma(gamma, config)
    yy = pi_gamma(xx)
        
    mean = np.trapezoid(xx * yy, xx)
    second_moment = np.trapezoid(xx**2 * yy, xx)
        
    return second_moment - mean**2

def mse_gamma_mean(gamma, config, samples):
    mn = true_mean_gamma(gamma, config)
    return (mn - np.mean(samples))**2

def mse_gamma_var(gamma, config, samples):
    vr = true_var_gamma(gamma, config)
    return (vr - np.mean(samples**2))**2

    
def prox_l1(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def myula_sampler(gamma, config: SamplerConfig, SEED=0):
    np.random.seed(SEED)
    _, _, beta, lam, n_samples, burn_in = config.get_params()
    G_prime = config.make_grad_G()
    
    L = np.abs(config.a) + 1/gamma  # Lipschitz constant 
    h = 1.0/L
    # h = gamma / (5*(gamma*L + 1))

    x = 0.0
    samples = []

    for _ in range(n_samples+burn_in):
        grad_G = G_prime(x)

        prox = prox_l1(x, gamma * lam)

        grad_R_gamma = (x - prox) / gamma

        noise = np.sqrt(2 * h / beta) * np.random.randn()

        x = x - h * (grad_G + grad_R_gamma) + noise

        samples.append(x)

    # return np.array(samples)
    return np.array(samples[burn_in:])
    

def hadamard_sampler(h, config: SamplerConfig, SEED=0):
    np.random.seed(SEED)
    a, b, beta, lam, n_samples, burn_in = config.get_params()
    G_prime = config.make_grad_G()

    samples = []

    # initial values
    u = 1.0
    v = 1.0

    for _ in range(n_samples + burn_in):
        
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
        
        samples.append(u * v)
    return np.array(samples[burn_in:])

def main_all(gamma=0.0031622776601683794, h=0.03162277660168379):

    config = SamplerConfig()
    sample_size = config.n_samples
    print('Sampling...')
    sample_gib = gibbs_sampler(config)
    print('Done Gibbs sampling')
    sample_myula = myula_sampler(gamma, config)
    print('Done MYULA sampling')
    sample_hadamard = hadamard_sampler(h, config)
    print('Done Hadamard sampling')

    names = ['Gibbs', 'MYULA', 'Hadamard']
    all_samples = [sample_gib, sample_myula, sample_hadamard]
    mn, mx = np.min(all_samples), np.max(all_samples)

    xx = np.linspace(mn - 1, mx + 1, sample_size)
    n_samples, burn_in = config.n_samples, config.burn_in
    pi_gamma = make_pi_gamma(gamma, config)

    fig, axs = config.plot_all_samples(all_samples, names)
    fig.suptitle(f'Comparison of Samplers. {burn_in:.0e} burn in, {n_samples:.0e} samples', fontsize=16)
    axs[1].plot(xx, pi_gamma(xx), color='green', linestyle='--', label=r'$\pi_\gamma$')
    axs[1].legend()
    fig.tight_layout()
    plt.savefig('Images/all_samplers.pdf', bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    # config=SamplerConfig()
    # config.posterior()
    main_all()

