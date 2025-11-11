import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
import emcee
import scipy.linalg as la

# Load data
data = np.loadtxt("pantheon_data_M.txt")
z_data_sn = data[:, 0]
mu_sn = data[:, 1]
cov_data = np.loadtxt("Pantheon_cov_matrix.cov")
cov_mat = cov_data.reshape(1701, 1701)
inverse_covar = la.inv(cov_mat)

# CPL dark energy equation of state
def wd(z, params):
    od0, H0, w0, wa = params
    return w0 + wa * z / (1 + z)

# Differential equations
def equation(z, variable, params):
    od, H, dl = variable
    od0, H0, w0, wa = params
    
    dotH = (-3 * (wd(z, params) * od + 1))
    
    eqd = 1 / (1 + z) * (3 * od * (1 + wd(z, params)) + dotH * od)
    eqH = (3 / (2 * (1 + z))) * H * (wd(z, params) * od + 1)
    eqdl = 1/(1+z)*dl + (1/H) * 2.99792458e5 * (1+z)

    return [eqd, eqH, eqdl]

# Distance modulus model
def mu_model(z, params):
    od0, H0, w0, wa = params
    
    sol = solve_ivp(lambda t, y: equation(t, y, params), 
                   [0, 3], 
                   [od0, H0, 0], 
                   t_eval=np.unique(z_data_sn),
                   method='RK45')
    
    dl_val = interp1d(sol.t, sol.y[2], kind='linear', fill_value="extrapolate")
    return 5*np.log10(dl_val(z)) + 25

# Statistical functions
def chisq(D, T, err):
    diff = D - T
    return np.dot(diff.T, np.dot(err, diff))

def log_prior(params):
    od0, H0, w0, wa = params
    if not (0.5 < od0 < 1):
        return -np.inf
    if not (40 < H0 < 99):
        return -np.inf
    if not (-1.5 < w0 < -0.3):
        return -np.inf
    if not (-2.0 < wa < 2.0):
        return -np.inf
    return 0

def log_prob(params):
    prior = log_prior(params)
    if prior == -np.inf:
        return -np.inf
    
    mu = mu_model(z_data_sn, params)
    if np.any(np.isinf(mu)):
        return -np.inf

    return -0.5 * chisq(mu_sn, mu, inverse_covar)

def aic(log_likelihood, ndim):
    return -2 * log_likelihood + 2 * ndim

def bic(log_likelihood, ndim, ndata):
    return -2 * log_likelihood + ndim * np.log(ndata)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # MCMC parameters (now in global scope)
    nwalker = 30
    ndim = 4
    niter = 5000
    dis = 500   # Burn-in
    th = 10     # Thinning
    
    # Initial positions
    p0 = np.random.normal(loc=[0.7, 70, -1.0, 0.0],
                         scale=[0.05, 5, 0.2, 0.5],
                         size=(nwalker, ndim))

    # Run MCMC
    ncpu = min(10, cpu_count())
    print(f"Using {ncpu} CPUs")

    with Pool(processes=ncpu) as pool:
        sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob, pool=pool)
        sampler.run_mcmc(p0, niter, progress=True)
    
    # Analysis
    chains = sampler.get_chain(flat=True, discard=dis, thin=th)
    samples = sampler.get_chain(discard=dis, thin=th, flat=True)

    # Parameter estimates
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    names = ['od0', 'H0', 'w0', 'wa']
    for i in range(ndim):
        median = percentiles[1, i]
        lower = median - percentiles[0, i]
        upper = percentiles[2, i] - median
        print(f"{names[i]} = {median:.3f} +{upper:.3f} -{lower:.3f}")

    # Corner plot
    fig = corner.corner(
        samples,
        labels=[r'$\Omega_d$', r'$H_0$', r'$w_0$', r'$w_a$'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    fig.savefig('SN_CPL_corner.pdf')
    plt.close()

    # Save results
    log_likelihood = np.mean(sampler.get_log_prob(discard=dis))
    with open("SN_results_CPL.txt", "w") as f:
        for i in range(ndim):
            f.write(f"{names[i]} = {percentiles[1,i]:.3f} +{percentiles[2,i]-percentiles[1,i]:.3f} -{percentiles[1,i]-percentiles[0,i]:.3f}\n")
        f.write(f"AIC = {aic(log_likelihood, ndim):.1f}\n")
        f.write(f"BIC = {bic(log_likelihood, ndim, len(z_data_sn)):.1f}\n")

        plt.figure(figsize=(8, 6))
    
    # Create redshift range for plotting
    z_plot = np.linspace(0, 2, 100)
    
    # Plot median w(z)
    median_params = percentiles[1,:]  # 50th percentile values
    w_median = wd(z_plot, median_params)
    plt.plot(z_plot, w_median, 'r-', lw=2, label='Median')
    
    # Plot uncertainty band (16th-84th percentiles)
    samples_w = np.array([wd(z_plot, p) for p in samples[np.random.choice(len(samples), 200)]]) # Subsample for efficiency
    w_lower = np.percentile(samples_w, 16, axis=0)
    w_upper = np.percentile(samples_w, 84, axis=0)
    plt.fill_between(z_plot, w_lower, w_upper, color='r', alpha=0.3, label='1σ uncertainty')
    
    # Plot ΛCDM reference line (w = -1)
    plt.axhline(-1, color='k', linestyle='--', label='ΛCDM (w=-1)')
    
    plt.xlabel('Redshift z')
    plt.ylabel('w(z)')
    plt.title('Dark Energy Equation of State Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('w_z_evolution.pdf')
    plt.close()
