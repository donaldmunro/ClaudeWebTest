"""
Flatland Diffusion Model
========================

A simple diffusion model for fixed-length bitstrings, inspired by the novel "Flatland"
where images are represented as bitstrings (1D sequences).

This implementation demonstrates:
1. Forward diffusion process (adding Gaussian noise)
2. Reverse diffusion process (denoising)
3. Connections to Stochastic Differential Equations (SDEs)

Mathematical Background:
-----------------------
The forward diffusion process follows an SDE (Ito form):
    dx_t = -½β(t)x_t dt + √β(t) dW_t

Where:
- x_t: data at time t
- β(t): noise schedule (variance schedule)
- W_t: Wiener process (Brownian motion)
- dt: infinitesimal time increment
- dW_t: infinitesimal Brownian increment

The reverse process (denoising) follows the reverse-time SDE:
    dx_t = [-½β(t)x_t - β(t)∇_x log p_t(x)] dt + √β(t) dW̃_t

Where ∇_x log p_t(x) is the score function (learned by neural networks in practice).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class FlatlandDiffusionModel:
    """
    Diffusion model for fixed-length bitstrings.

    Bitstrings are mapped to continuous space {0,1} -> {-1,+1} before diffusion.
    This allows us to apply Gaussian noise while maintaining a meaningful representation.
    """

    def __init__(self, bitstring_length: int, num_timesteps: int = 100):
        """
        Initialize the Flatland diffusion model.

        Args:
            bitstring_length: Length of the bitstrings (dimension of Flatland images)
            num_timesteps: Number of diffusion steps (discretization of continuous time)
        """
        self.d = bitstring_length  # Dimension (length of bitstring)
        self.T = num_timesteps     # Number of timesteps

        # Define noise schedule β(t)
        # This is β(t) from the SDE: controls how quickly noise is added
        # Linear schedule from β_start to β_end
        self.beta = np.linspace(0.0001, 0.02, num_timesteps)

        # Precompute useful quantities for efficient sampling
        # α_t = 1 - β_t (used in discrete-time formulation)
        self.alpha = 1.0 - self.beta

        # ᾱ_t = ∏_{s=1}^t α_s (cumulative product)
        # This allows us to sample x_t directly from x_0 without iterating
        # Derived from solving the SDE analytically
        self.alpha_bar = np.cumprod(self.alpha)

        # For the reverse process
        self.alpha_bar_prev = np.concatenate([[1.0], self.alpha_bar[:-1]])

    def bitstring_to_continuous(self, bitstring: np.ndarray) -> np.ndarray:
        """
        Convert bitstring {0,1}^d to continuous representation {-1,+1}^d.

        This mapping: 0 -> -1, 1 -> +1 is chosen because:
        - It's symmetric around 0 (good for Gaussian noise)
        - Easy to threshold back: sign(x) -> {-1,+1} -> {0,1}
        """
        return 2.0 * bitstring - 1.0

    def continuous_to_bitstring(self, continuous: np.ndarray) -> np.ndarray:
        """
        Convert continuous representation back to bitstring.
        Uses thresholding at 0: negative -> 0, positive -> 1
        """
        return (continuous > 0).astype(int)

    def forward_diffusion_step(self, x: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single step of forward diffusion process.

        Implements the discrete-time version of the SDE:
            x_t = √(1-β_t) * x_{t-1} + √β_t * ε

        where ε ~ N(0, I) is Gaussian noise.

        This comes from the Euler-Maruyama discretization of the SDE:
            dx = -½β(t)x dt + √β(t) dW

        Args:
            x: Current state (continuous representation)
            t: Timestep index

        Returns:
            (noisy_x, noise): The noised data and the noise that was added
        """
        # Sample Gaussian noise: dW in the SDE formulation
        noise = np.random.randn(*x.shape)

        # Discrete approximation of the SDE
        # √α_t corresponds to exp(-½β_t dt) ≈ 1 - ½β_t in continuous time
        sqrt_alpha = np.sqrt(self.alpha[t])
        sqrt_one_minus_alpha = np.sqrt(1.0 - self.alpha[t])

        # x_t = √α_t * x_{t-1} + √(1-α_t) * ε
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise

        return noisy_x, noise

    def forward_diffusion_direct(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct sampling of x_t from x_0 (without iterating through all steps).

        This is the closed-form solution to the SDE:
            x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        where ᾱ_t = ∏_{s=1}^t α_s

        This comes from the Ornstein-Uhlenbeck process solution, where
        the forward SDE has an analytical solution.

        Args:
            x_0: Original clean data
            t: Target timestep

        Returns:
            (x_t, noise): Data at timestep t and the total noise added
        """
        # Sample noise from standard Gaussian
        noise = np.random.randn(*x_0.shape)

        # Analytical solution to the forward SDE
        sqrt_alpha_bar = np.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar[t])

        # Apply the closed-form transformation
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def reverse_diffusion_step(self, x_t: np.ndarray, t: int,
                               score_fn: Optional[callable] = None) -> np.ndarray:
        """
        Single step of reverse diffusion (denoising).

        Implements the reverse-time SDE:
            dx = [½β(t)x + β(t)∇_x log p_t(x)] dt + √β(t) dW̃

        In practice, neural networks learn the score ∇_x log p_t(x) or equivalently
        the noise ε. Here we use a simple approximation for demonstration.

        Args:
            x_t: Current noisy state at timestep t
            t: Current timestep
            score_fn: Optional function to estimate the score/noise

        Returns:
            x_{t-1}: Less noisy state at timestep t-1
        """
        if t == 0:
            return x_t

        # Get variance schedule values
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar_prev[t]

        # Simple denoising: assume data was originally near {-1, +1}
        # In real diffusion models, this would be a learned neural network
        if score_fn is None:
            # Naive approach: project towards nearest valid point
            predicted_x0 = np.clip(x_t / np.sqrt(alpha_bar_t), -1.5, 1.5)
            predicted_x0 = np.sign(predicted_x0)  # Force towards {-1, +1}
        else:
            predicted_x0 = score_fn(x_t, t)

        # Compute mean of reverse distribution
        # This follows from the reverse SDE formula
        # μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
        coef1 = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t) * predicted_x0
        mean += np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * x_t

        # Add noise (except at final step)
        # This is the √β(t) dW̃ term in the reverse SDE
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

        if t > 0:
            noise = np.random.randn(*x_t.shape)
            x_prev = mean + np.sqrt(variance) * noise
        else:
            x_prev = mean

        return x_prev

    def sample(self, batch_size: int = 1,
               score_fn: Optional[callable] = None) -> np.ndarray:
        """
        Generate new bitstrings by sampling from the reverse diffusion process.

        Process:
        1. Start from pure Gaussian noise (t=T)
        2. Iteratively denoise following the reverse SDE
        3. Convert final continuous values to bitstrings

        Args:
            batch_size: Number of samples to generate
            score_fn: Optional learned score function

        Returns:
            Generated bitstrings
        """
        # Start from pure noise: x_T ~ N(0, I)
        # This is the stationary distribution of the forward SDE
        x = np.random.randn(batch_size, self.d)

        # Reverse diffusion: iterate backwards through time
        for t in reversed(range(self.T)):
            x = self.reverse_diffusion_step(x, t, score_fn)

        # Convert continuous representation back to bitstrings
        bitstrings = self.continuous_to_bitstring(x)

        return bitstrings

    def demonstrate_forward_process(self, bitstring: np.ndarray,
                                   timesteps_to_show: list = None) -> dict:
        """
        Visualize the forward diffusion process on a single bitstring.

        Args:
            bitstring: Input bitstring to diffuse
            timesteps_to_show: Specific timesteps to capture

        Returns:
            Dictionary containing the diffusion trajectory
        """
        if timesteps_to_show is None:
            timesteps_to_show = [0, self.T//4, self.T//2, 3*self.T//4, self.T-1]

        # Convert to continuous representation
        x_0 = self.bitstring_to_continuous(bitstring)

        trajectory = {'timesteps': [], 'continuous': [], 'bitstrings': []}

        for t in timesteps_to_show:
            x_t, noise = self.forward_diffusion_direct(x_0, t)

            trajectory['timesteps'].append(t)
            trajectory['continuous'].append(x_t.copy())
            trajectory['bitstrings'].append(self.continuous_to_bitstring(x_t))

        return trajectory


def visualize_diffusion_process(model: FlatlandDiffusionModel,
                                bitstring: np.ndarray,
                                save_path: str = 'flatland_diffusion_demo.png'):
    """
    Create a visualization of the diffusion process.

    Shows:
    1. Forward process: clean bitstring -> noise
    2. Reverse process: noise -> reconstructed bitstring
    3. Continuous representation evolution
    """
    print("Visualizing diffusion process...")

    # Forward process
    forward_traj = model.demonstrate_forward_process(bitstring)

    # Reverse process (sampling)
    # Start from the final noisy state
    x_T = forward_traj['continuous'][-1]
    reverse_trajectory = [x_T.copy()]

    for t in reversed(range(model.T)):
        x_t = model.reverse_diffusion_step(reverse_trajectory[-1], t)
        reverse_trajectory.append(x_t.copy())

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Forward process - bitstrings
    ax = axes[0]
    forward_bits = np.array([forward_traj['bitstrings'][i]
                            for i in range(len(forward_traj['timesteps']))])
    im1 = ax.imshow(forward_bits, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_ylabel('Timestep', fontsize=12)
    ax.set_xlabel('Bit Position', fontsize=12)
    ax.set_title('Forward Diffusion: Clean Bitstring → Noise (Discrete View)',
                 fontsize=14, fontweight='bold')
    timestep_labels = [f"t={t}" for t in forward_traj['timesteps']]
    ax.set_yticks(range(len(timestep_labels)))
    ax.set_yticklabels(timestep_labels)
    plt.colorbar(im1, ax=ax, label='Bit Value')

    # Plot 2: Forward process - continuous representation
    ax = axes[1]
    forward_cont = np.array([forward_traj['continuous'][i]
                            for i in range(len(forward_traj['timesteps']))])
    im2 = ax.imshow(forward_cont, cmap='RdBu_r', aspect='auto',
                    interpolation='nearest', vmin=-3, vmax=3)
    ax.set_ylabel('Timestep', fontsize=12)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_title('Forward Diffusion: Continuous Representation (SDE Solution)',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(timestep_labels)))
    ax.set_yticklabels(timestep_labels)
    plt.colorbar(im2, ax=ax, label='Continuous Value')

    # Plot 3: Reverse process sampling
    ax = axes[2]
    # Show selected steps from reverse process
    n_steps = len(forward_traj['timesteps'])
    reverse_indices = np.linspace(0, len(reverse_trajectory)-1, n_steps).astype(int)
    reverse_show = np.array([reverse_trajectory[i] for i in reverse_indices])
    im3 = ax.imshow(reverse_show, cmap='RdBu_r', aspect='auto',
                    interpolation='nearest', vmin=-3, vmax=3)
    ax.set_ylabel('Timestep', fontsize=12)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_title('Reverse Diffusion: Noise → Reconstructed (Following Reverse SDE)',
                 fontsize=14, fontweight='bold')
    reverse_labels = [f"t={model.T - i*model.T//(n_steps-1)}" for i in range(n_steps)]
    ax.set_yticks(range(len(reverse_labels)))
    ax.set_yticklabels(reverse_labels)
    plt.colorbar(im3, ax=ax, label='Continuous Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

    return fig


def main():
    """
    Demonstration of the Flatland Diffusion Model.
    """
    print("="*70)
    print("FLATLAND DIFFUSION MODEL DEMONSTRATION")
    print("="*70)
    print()

    # Create a sample bitstring (a simple pattern)
    # In Flatland, this could represent a 1D "image"
    bitstring_length = 32

    # Create an interesting pattern: alternating blocks
    pattern = np.array([1,1,1,1,0,0,0,0] * 4, dtype=int)

    print(f"Original Flatland Image (bitstring of length {bitstring_length}):")
    print(f"  {''.join(map(str, pattern))}")
    print()

    # Initialize the diffusion model
    num_timesteps = 100
    model = FlatlandDiffusionModel(bitstring_length, num_timesteps)

    print(f"Diffusion Model Parameters:")
    print(f"  - Bitstring length (dimension): {bitstring_length}")
    print(f"  - Number of timesteps: {num_timesteps}")
    print(f"  - Beta schedule: linear from {model.beta[0]:.4f} to {model.beta[-1]:.4f}")
    print()

    # Demonstrate forward diffusion
    print("-"*70)
    print("FORWARD DIFFUSION PROCESS (Adding Gaussian Noise)")
    print("-"*70)
    print()

    trajectory = model.demonstrate_forward_process(pattern)

    for i, t in enumerate(trajectory['timesteps']):
        bits = trajectory['bitstrings'][i]
        cont = trajectory['continuous'][i]
        print(f"Timestep t={t:3d}:")
        print(f"  Bitstring:  {''.join(map(str, bits))}")
        print(f"  Continuous: mean={cont.mean():6.3f}, std={cont.std():6.3f}")
        print()

    # Demonstrate reverse diffusion (sampling)
    print("-"*70)
    print("REVERSE DIFFUSION PROCESS (Removing Noise / Sampling)")
    print("-"*70)
    print()

    num_samples = 5
    samples = model.sample(batch_size=num_samples)

    print(f"Generated {num_samples} new Flatland images:")
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}: {''.join(map(str, sample))}")
    print()

    # Compute Hamming distance to original pattern
    print("Hamming distances from original pattern:")
    for i, sample in enumerate(samples):
        distance = np.sum(sample != pattern)
        print(f"  Sample {i+1}: {distance}/{bitstring_length} bits different")
    print()

    # Create visualization
    print("-"*70)
    print("Creating visualization...")
    visualize_diffusion_process(model, pattern)

    print()
    print("="*70)
    print("SDE Connection Summary:")
    print("="*70)
    print("""
The diffusion model implements discretized versions of SDEs:

1. FORWARD SDE (continuous time):
   dx_t = -½β(t)x_t dt + √β(t) dW_t

   Discretized: x_t = √α_t * x_{t-1} + √(1-α_t) * ε

2. REVERSE SDE (continuous time):
   dx_t = [½β(t)x_t + β(t)∇_x log p_t(x)] dt + √β(t) dW̃_t

   The score function ∇_x log p_t(x) guides denoising.

3. The α and ᾱ terms come from solving the forward SDE analytically,
   allowing direct sampling at any timestep t without iteration.

4. The noise schedule β(t) controls the diffusion rate - how quickly
   information is destroyed (forward) or reconstructed (reverse).
""")

    print("Demonstration complete!")


if __name__ == "__main__":
    main()
