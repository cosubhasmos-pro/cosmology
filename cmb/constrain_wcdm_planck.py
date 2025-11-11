import os
import numpy as np
import camb
from cobaya.run import run
from cobaya.yaml import yaml_load
import getdist
from getdist import plots
import matplotlib.pyplot as plt

# Define the Cobaya input dictionary for wCDM model
info = {
    "likelihood": {
        "planck_2018_lowl.TT": None,  # Low-ell TT likelihood
        "planck_2018_lowl.EE": None,  # Low-ell EE likelihood
        "planck_2018_highl_plik.TTTEEE": None,  # High-ell TT, TE, EE likelihood
        "planck_2018_lensing.clik": None  # CMB lensing likelihood
    },
    "theory": {
        "camb": {
            "extra_args": {
                "num_massive_neutrinos": 1,
                "halofit_version": "mead",
                "lmax": 2500,
                "dark_energy_model": "fluid",  # Constant w dark energy model
                "lens_potential_accuracy": 1
            }
        }
    },
    "params": {
        # Cosmological parameters with priors and reference values
        "ombh2": {"prior": {"min": 0.005, "max": 0.1}, "ref": 0.0224, "proposal": 0.0001},
        "omch2": {"prior": {"min": 0.001, "max": 0.99}, "ref": 0.119, "proposal": 0.001},
        "H0": {"prior": {"min": 40, "max": 100}, "ref": 67.0, "proposal": 2.0},
        "As": {"prior": {"min": 1e-9, "max": 4e-9}, "ref": 2.1e-9, "proposal": 0.05e-9},
        "ns": {"prior": {"min": 0.8, "max": 1.2}, "ref": 0.965, "proposal": 0.004},
        "tau": {"prior": {"min": 0.01, "max": 0.2}, "ref": 0.054, "proposal": 0.008},
        # Dark energy equation of state (constant w)
        "w": {"prior": {"min": -2.0, "max": -0.3}, "ref": -1.0, "proposal": 0.05}
    },
    "sampler": {
        "mcmc": {
            "Rminus1_stop": 0.01,  # Stop when Gelman-Rubin R-1 < 0.01
            "Rminus1_cl_stop": 0.2,  # Convergence for confidence limits
            "max_tries": 10000,  # Maximum MCMC steps
            "covmat": None,  # Auto-determine covariance matrix
            "proposal_scale": 1.9  # Scale for proposal distribution
        }
    },
    "output": "chains/wcdm_planck"  # Output directory for MCMC chains
}

# Set path to Planck likelihood data (update with your path)
info["packages_path"] = os.path.expanduser("./planck_data")

# Run the MCMC sampler
print("Starting MCMC sampling...")
updated_info, sampler = run(info)
print("MCMC sampling completed.")

# Load the MCMC chains
samples = sampler.products()["sample"]

# Analyze and plot results with GetDist
g = plots.get_subplot_plotter(subplot_size=3)
g.settings.title_limit_fontsize = 12
params_to_plot = ["ombh2", "omch2", "H0", "w"]
g.triangle_plot(samples, params_to_plot, filled=True, title_limit=1)
g.export("wcdm_planck_triangle.pdf")
print("Triangle plot saved as wcdm_planck_triangle.pdf")

# Plot 1D posterior for w
plt.figure(figsize=(6, 4))
g.plot_1d(samples, "w")
plt.title("Posterior for Dark Energy Equation of State (w)")
plt.xlabel("w")
plt.ylabel("Probability Density")
plt.savefig("wcdm_planck_w_posterior.pdf")
plt.close()
print("1D posterior for w saved as wcdm_planck_w_posterior.pdf")

# Save parameter constraints to a text file
with open("wcdm_planck_constraints.txt", "w") as f:
    f.write("Parameter Constraints from Planck 2018 (wCDM)\n")
    f.write("----------------------------------------\n")
    for param in params_to_plot:
        mean = samples[param].mean()
        std = samples[param].std()
        f.write(f"{param}: {mean:.4f} ± {std:.4f}\n")
print("Constraints saved to wcdm_planck_constraints.txt")