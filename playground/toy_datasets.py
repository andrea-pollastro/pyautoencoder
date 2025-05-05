import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from scipy.stats import multivariate_normal

def generate_nonlinear_1D(z: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
    """
    Nonlinear generative function mapping latent variables z in R^2 to observations x in R^1.

    The transformation is defined as:
        x = sin(z_1) + cos(z_2) + epsilon,
    where epsilon ~ N(0, noise_std^2) is Gaussian noise with standard deviation `noise_std`.

    This creates a smooth, nonlinear mapping from 2D latent space to 1D observations.
    The resulting posterior p(z | x) is generally multimodal and useful for evaluating
    inference quality in models like VAEs.

    Args:
        z (np.ndarray): Latent samples of shape (N, 2).
        noise_std (float): Standard deviation of the Gaussian observation noise.

    Returns:
        np.ndarray: Observed values x of shape (N, 1).
    """
    x = np.sin(z[:, 0]) + np.cos(z[:, 1])
    noise = np.random.normal(0, noise_std, size=z.shape[0])
    return (x + noise).reshape(-1, 1)

def generate_nonlinear_2D(z: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
    """
    Nonlinear generative function mapping latent variables z in R^2 to observations x in R^2.

    The transformation is defined component-wise as:
        x_1 = sin(z_1) + cos(z_2)
        x_2 = sin(2 * z_1) - cos(0.5 * z_2)
    and both components are perturbed by independent Gaussian noise:
        epsilon ~ N(0, noise_std^2)

    This function introduces complex nonlinearities, making the posterior p(z | x)
    highly non-Gaussian and multimodal. It is well-suited for testing inference models
    under more realistic, nonlinear generative conditions.

    Args:
        z (np.ndarray): Latent samples of shape (N, 2).
        noise_std (float): Standard deviation of the Gaussian observation noise.

    Returns:
        np.ndarray: Observed values x of shape (N, 2).
    """
    x1 = np.sin(z[:, 0]) + np.cos(z[:, 1])
    x2 = np.sin(2 * z[:, 0]) - np.cos(0.5 * z[:, 1])
    noise = np.random.normal(0, noise_std, size=(z.shape[0], 2))
    return np.stack([x1, x2], axis=1) + noise

def make_mixture_gaussian(n_samples: int = 1000,
                          disposition: str = 'random', 
                          n_components: int = 10, 
                          cov_scaling: float = 0.02,
                          spacing: float = .5,
                          n_components_per_axis: int = 2,
                          radius: float = 1, 
                          generative_fn: Callable = generate_nonlinear_1D,
                          generative_noise_std: float = 0.1,
                          random_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates samples from a 2D latent mixture of Gaussians and applies a nonlinear generative function.

    Args:
        n_samples (int): Number of latent samples.
        disposition (str): One of {'random', 'grid', 'circle'}.
        n_components (int): Number of mixture components.
        cov_scaling (float): Scaling factor for component covariances.
        spacing (float): Distance between components (for grid/random).
        n_components_per_axis (int): Components per axis (for grid).
        radius (float): Radius of the circle (for 'circle' disposition).
        generative_fn (Callable): Function mapping z to x.
        generative_noise_std (float): Standard deviation of noise in the generative process.
        random_seed (int): Seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - x: Generated observations (N, d)
            - z: Latent samples (N, 2)
            - centers: Component centers (K, 2)
            - cov: Covariance matrix (2, 2)
            - component_ids: Index of component for each sample (N,)
    """
    if disposition not in ['random', 'grid', 'circle']:
        raise ValueError(f"Unknown disposition: '{disposition}'. Choose 'random', 'circle' or 'grid'.")
    np.random.seed(random_seed)

    if disposition == 'grid':
        lin = np.linspace(-spacing, spacing, n_components_per_axis)
        mesh_x, mesh_y = np.meshgrid(lin, lin)
        centers = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)  # shape: (K, 2)
        n_components = centers.shape[0]

    elif disposition == 'circle':
        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    elif disposition == 'random':
        centers = np.random.uniform(low=-spacing, high=spacing, size=(n_components, 2))

    cov = cov_scaling * np.eye(2)
    component_ids = np.random.randint(0, n_components, size=n_samples)
    z = np.array([np.random.multivariate_normal(centers[i], cov) for i in component_ids])
    x = generative_fn(z, noise_std=generative_noise_std)

    return x, z, centers, cov, component_ids

def compute_true_posterior(x0: np.ndarray,
                           z: np.ndarray, 
                           centers: np.ndarray,
                           cov: np.ndarray,
                           generative_fn: Callable = generate_nonlinear_1D,
                           generative_noise_std: float = 0.1,
                           grid_size: int = 100, 
                           h: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the true posterior density p(z | x) on a 2D grid using Bayes' rule.

    Args:
        x0 (np.ndarray): Observation vector of shape (d,).
        z (np.ndarray): Latent samples used for grid bounds (N, 2).
        centers (np.ndarray): Mixture component means (K, 2).
        cov (np.ndarray): Shared covariance matrix for components (2, 2).
        generative_fn (Callable): Generative function used to compute p(x|z).
        generative_noise_std (float): Noise standard deviation in the likelihood model.
        grid_size (int): Number of grid points along each axis.
        h (float): Margin added to min/max of z grid range.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Z1, Z2: Grid coordinates
            - p_z_given_x: Posterior density p(z | x) over the grid
    """
    z1 = np.linspace(z[:, 0].min() - h, z[:, 0].max() + h, grid_size)
    z2 = np.linspace(z[:, 1].min() - h, z[:, 1].max() + h, grid_size)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.stack([Z1.ravel(), Z2.ravel()], axis=-1)

    ## Bayes theorem, p(z|x) ∝ p(x|z)p(z)
    # p(z)
    p_z = np.zeros(Z.shape[0])
    for mu in centers:
        p_z += multivariate_normal.pdf(Z, mean=mu, cov=cov)
    p_z /= len(centers)

    # log p(x|z)
    diff = generative_fn(Z, noise_std=0.0) - x0
    log_p_x_given_z = -0.5 * np.sum((diff / generative_noise_std) ** 2, axis=1) \
                      - x0.shape[0] * np.log(np.sqrt(2 * np.pi) * generative_noise_std)


    # p(z|x)
    log_p_z_given_x = np.log(p_z + 1e-12) + log_p_x_given_z          # add small value to avoid log(0)
    p_z_given_x = np.exp(log_p_z_given_x - np.max(log_p_z_given_x))  # for numerical stability
    p_z_given_x /= np.sum(p_z_given_x)
    p_z_given_x = p_z_given_x.reshape(grid_size, grid_size)
    p_z_given_x = p_z_given_x / p_z_given_x.sum()

    return Z1, Z2, p_z_given_x

def plot_latent_and_true_posterior(z: np.ndarray,
                                   x0: np.ndarray,
                                   x: np.ndarray,
                                   Z1: np.ndarray,
                                   Z2: np.ndarray,
                                   p_z_given_x: np.ndarray):
    """
    Plots latent samples colored by each dimension of x and the true posterior p(z | x).

    Args:
        z (np.ndarray): Latent samples of shape (N, 2).
        x0 (np.ndarray): Observation vector used for posterior (d,).
        x (np.ndarray): All generated outputs x of shape (N, d).
        Z1 (np.ndarray): Meshgrid of z1 for plotting.
        Z2 (np.ndarray): Meshgrid of z2 for plotting.
        p_z_given_x (np.ndarray): Posterior density p(z | x) over the grid.
    """
    d = x.shape[1]
    fig, axs = plt.subplots(1, d + 1, figsize=(5 * (d + 1), 4), sharey=True)

    for i in range(d):
        sc = axs[i].scatter(z[:, 0], z[:, 1], c=x[:, i], cmap='coolwarm', s=2, alpha=0.6)
        plt.colorbar(sc, ax=axs[i], label=f'$x_{{{i+1}}}$')
        axs[i].set_title(f'Latent colored by $x_{{{i+1}}}$', fontsize=12)
        axs[i].set_xlabel('$z_1$', fontsize=10)
        axs[i].set_ylabel('$z_2$', fontsize=10)
        axs[i].axis("equal")
        axs[i].grid(alpha=0.2)

    cb = axs[-1].contourf(Z1, Z2, p_z_given_x, levels=50, cmap='viridis')
    plt.colorbar(cb, ax=axs[-1], label='Density')
    axs[-1].set_title(f'True Posterior $p(z \\mid x=[{",".join(str(np.round(i, 3)) for i in x0)}])$', fontsize=12)
    axs[-1].set_xlabel('$z_1$', fontsize=10)
    axs[-1].set_ylabel('$z_2$', fontsize=10)
    axs[-1].axis("equal")
    axs[-1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
