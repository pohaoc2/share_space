"""
Visualization of Fréchet Distance Permutation Test
Creates an animated GIF showing the distribution of permuted distances
"""
from typing import Tuple, Optional
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from scipy import linalg

warnings.filterwarnings('ignore')


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """Fréchet distance between two Gaussians (vectors: mu, cov)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)
    diff = mu1 - mu2
    
    # sqrt of product might be numerical; add eps to diag
    try:
        cov_prod_sqrt = linalg.sqrtm(cov1.dot(cov2))
    except Exception:
        cov_prod_sqrt = None

    if cov_prod_sqrt is not None and np.iscomplexobj(cov_prod_sqrt):
        # numerical noise -> take real part
        cov_prod_sqrt = cov_prod_sqrt.real

    # If still has NaNs or so, regularize covariances
    if cov_prod_sqrt is None or not np.isfinite(cov_prod_sqrt).all():
        cov1 += np.eye(cov1.shape[0]) * eps
        cov2 += np.eye(cov2.shape[0]) * eps
        cov_prod_sqrt = linalg.sqrtm(cov1.dot(cov2)).real

    tr_term = np.trace(cov1 + cov2 - 2.0 * cov_prod_sqrt)
    return float(diff.dot(diff) + tr_term)


def compute_frechet_from_samples(X, Y, eps=1e-6):
    """
    X, Y: arrays shape (N, D) and (M, D)
    """
    muX = np.mean(X, axis=0)
    muY = np.mean(Y, axis=0)
    covX = np.cov(X, rowvar=False)
    covY = np.cov(Y, rowvar=False)
    return frechet_distance(muX, covX, muY, covY, eps=eps)


def draw_break_marks(axes):
    """
    Draw break marks between axes in a broken axis plot.
    
    Parameters:
    -----------
    axes : list of matplotlib.axes.Axes
        List of axes to draw break marks between
    """
    d = 0.015  # Size of break marks
    
    for i in range(len(axes) - 1):
        ax_left = axes[i]
        ax_right = axes[i + 1]
        
        # Draw break marks on left axis (right side)
        kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        
        # Draw break marks on right axis (left side)
        kwargs = dict(transform=ax_right.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax_right.plot((-d, +d), (-d, +d), **kwargs)
        ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)


def setup_broken_axis_histogram(fig, pos, x_ranges=[(-0.1, 0.5), (3, 3.5)]):
    """
    Create a broken axis setup for histogram using subplots.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to add subplots to
    pos : tuple
        Position for the broken axis subplot (left, bottom, width, height)
    x_ranges : list of tuples
        List of (x_min, x_max) tuples for each range to display
    
    Returns:
    --------
    axes : list of matplotlib.axes.Axes
        List of axes for each range
    """
    from matplotlib import gridspec
    
    n_ranges = len(x_ranges)
    left, bottom, width, height = pos
    
    # Calculate width ratios based on range sizes
    range_sizes = [xmax-xmin for xmin, xmax in x_ranges]
    total_size = sum(range_sizes)
    width_ratios = [s/total_size for s in range_sizes]
    
    # Create gridspec for broken axis
    gs = gridspec.GridSpec(1, n_ranges, 
                          left=left, bottom=bottom, 
                          right=left+width, top=bottom+height,
                          width_ratios=width_ratios, wspace=0.05)
    
    axes = []
    for i, (xmin, xmax) in enumerate(x_ranges):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(xmin, xmax)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.tick_params(left=False, labelleft=False)
        if i < n_ranges - 1:
            ax.spines['right'].set_visible(False)
        axes.append(ax)
    
    # Add break marks
    draw_break_marks(axes)
    
    return axes


def frechet_permutation_test_animated(
    X, 
    Y, 
    n_perms=200, 
    seed=0,
    save_dir=None,
    fps=10,
    figsize=(8, 3.75),
    n_bins=30
):
    """
    Conduct Fréchet permutation test and create animated visualization.
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Sample arrays of shape (N, D) and (M, D)
    n_perms : int
        Number of permutations
    seed : int
        Random seed for reproducibility
    save_dir : str
        Path to save the animated GIF
    fps : int
        Frames per second for the animation
    figsize : tuple
        Figure size (width, height)
    n_bins : int
        Number of bins for histogram
    
    Returns:
    --------
    obs_distance : float
        Observed Fréchet distance
    pval : float
        Permutation test p-value
    """
    np.random.seed(seed)
    combined = np.vstack([X, Y])
    n = X.shape[0]
    
    # Compute observed distance
    obs_distance = compute_frechet_from_samples(X, Y)
    
    # Store all permutation distances
    perm_distances = []
    current_A = None
    current_B = None
    
    # Setup figure with broken axis for histogram
    fig = plt.figure(figsize=figsize)
    x_ranges = [(-0.1, 1)]
    
    # Create broken axis for histogram (left side, takes ~45% of width) # left, bottom, width, height
    hist_axes = setup_broken_axis_histogram(fig, pos=(0.1, 0.15, 0.38, 0.75), x_ranges=x_ranges)
    
    # Right plot for scatter (takes ~45% of width)
    ax2 = fig.add_axes([0.58, 0.15, 0.38, 0.75])
    
    def init():
        for ax in hist_axes:
            ax.clear()
        ax2.clear()
        # Ensure y-axis is hidden for second histogram
        if len(hist_axes) > 1:
            hist_axes[1].spines['left'].set_visible(False)
            hist_axes[1].set_yticks([])
            hist_axes[1].tick_params(left=False, labelleft=False)
        return []
    
    def update(frame):
        nonlocal current_A, current_B
        if frame < n_perms:
            # Perform permutation
            perm = np.random.permutation(combined.shape[0])
            A = combined[perm[:n]]
            B = combined[perm[n:]]
            val = compute_frechet_from_samples(A, B)
            perm_distances.append(val)
            current_A = A
            current_B = B
        
        # Clear axes
        for ax in hist_axes:
            ax.clear()
        ax2.clear()
        
        # Left plot: Histogram of permutation distances with broken axis
        if len(perm_distances) > 0:
            # Get max frequency for consistent y-axis
            all_counts = []
            for xmin, xmax in x_ranges:
                range_data = [d for d in perm_distances if xmin <= d <= xmax]
                if len(range_data) > 0:
                    counts, _ = np.histogram(range_data, bins=n_bins, range=(xmin, xmax))
                    all_counts.extend(counts)
            max_freq = max(all_counts) if all_counts else 1
            
            # Filter data for each range and plot
            for i, (xmin, xmax) in enumerate(x_ranges):
                range_data = [d for d in perm_distances if xmin <= d <= xmax]
                if len(range_data) > 0:
                    hist_axes[i].hist(range_data, bins=n_bins, range=(xmin, xmax), 
                            alpha=0.7, color='skyblue', edgecolor='black', density=False)
                    hist_axes[i].set_ylim(0, max_freq * 1.1)
                if xmin <= obs_distance <= xmax:
                    hist_axes[i].axvline(obs_distance, color='red', linestyle='--', 
                                       linewidth=2.5, label=f'Observed: {obs_distance:.3f}')
                hist_axes[i].set_xlim(xmin, xmax)
            
            # Set labels and title
            hist_axes[0].set_ylabel('Frequency', fontsize=12)
            hist_axes[-1].set_xlabel('Fréchet Distance', fontsize=12)#, x=-0.15)
            hist_axes[0].set_title(f'Permutation (n={len(perm_distances)}/{n_perms})', 
                         fontsize=12, fontweight='bold')#, x=0.9)
            
            # Ensure y-axis is hidden for second histogram
            if len(hist_axes) > 1:
                hist_axes[1].spines['left'].set_visible(False)
                hist_axes[1].set_yticks([])
                hist_axes[1].tick_params(left=False, labelleft=False)
            
            # Redraw break marks (they get cleared with ax.clear())
            draw_break_marks(hist_axes)
        
        # Right plot: Scatter plot of A and B
        if current_A is not None and current_B is not None:
            # Use first two dimensions for scatter plot
            if current_A.shape[1] >= 2:
                ax2.scatter(current_A[:, 0], current_A[:, 1], s=50,
                            facecolors='none', edgecolors='green', marker='^', linewidths=1.5,
                            label='Fuse reconstructed')
                ax2.scatter(current_B[:, 0], current_B[:, 1], s=50, marker='s',
                            facecolors='none', edgecolors='#993F71', linewidths=1.5,
                            label='Experimental')
                ax2.set_xlabel('PC1', fontsize=12)
                ax2.set_ylabel('PC2', fontsize=12)
            
            ax2.set_title(f'PCA Samples (n={len(perm_distances)}/{n_perms})', 
                         fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_perms + 10, 
                        init_func=init, blit=False, repeat=False)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(Path(save_dir) / 'frechet_permutation_test.gif', writer=writer)
    plt.close()
    
    # Calculate final p-value
    count = sum(d >= obs_distance for d in perm_distances)
    pval = (count + 1) / (n_perms + 1)
    
    print(f"\n{'='*60}")
    print(f"Fréchet Distance Permutation Test Results")
    print(f"{'='*60}")
    print(f"Observed Fréchet Distance: {obs_distance:.6f}")
    print(f"Number of Permutations: {n_perms}")
    print(f"Permutations ≥ Observed: {count}")
    print(f"p-value: {pval:.6f}")
    print(f"Significant at α=0.05: {'Yes' if pval < 0.05 else 'No'}")
    print(f"{'='*60}")
    print(f"\nAnimation saved to: {Path(save_dir) / 'frechet_permutation_test.gif'}")
    
    return obs_distance, pval


def create_static_summary_plot(
    X,
    Y,
    n_perms=200,
    seed=0,
    save_dir=None,
    figsize=(7, 3.75),
    n_bins=30
):
    """
    Create a static summary plot showing the final results.
    """
    np.random.seed(seed)
    combined = np.vstack([X, Y])
    n = X.shape[0]
    
    # Compute observed distance
    obs_distance = compute_frechet_from_samples(X, Y)
    
    # Perform all permutations
    perm_distances = []
    for _ in range(n_perms):
        perm = np.random.permutation(combined.shape[0])
        A = combined[perm[:n]]
        B = combined[perm[n:]]
        val = compute_frechet_from_samples(A, B)
        perm_distances.append(val)
    
    # Calculate p-value
    count = sum(d >= obs_distance for d in perm_distances)
    pval = (count + 1) / (n_perms + 1)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    x_ranges = [(-0.1, 3)]
    
    # Create broken axis for histogram (left side, takes ~45% of width) # left, bottom, width, height
    hist_axes = setup_broken_axis_histogram(fig, pos=(0.1, 0.15, 0.38, 0.75), x_ranges=x_ranges)
    
    # Right plot for scatter (takes ~45% of width)
    ax2 = fig.add_axes([0.58, 0.15, 0.38, 0.75])
    
    # Histogram with broken axis
    all_counts = []
    for xmin, xmax in x_ranges:
        range_data = [d for d in perm_distances if xmin <= d <= xmax]
        if len(range_data) > 0:
            counts, _ = np.histogram(range_data, bins=n_bins, range=(xmin, xmax))
            all_counts.extend(counts)
    max_freq = max(all_counts) if all_counts else 1
    
    for i, (xmin, xmax) in enumerate(x_ranges):
        range_data = [d for d in perm_distances if xmin <= d <= xmax]
        if len(range_data) > 0:
            hist_axes[i].hist(range_data, bins=n_bins, range=(xmin, xmax),
                    alpha=0.7, color='skyblue', edgecolor='black', density=False)
            hist_axes[i].set_ylim(0, max_freq * 1.1)
        if xmin <= obs_distance <= xmax:
            hist_axes[i].axvline(obs_distance, color='red', linestyle='--', 
                               linewidth=2.5, label=f'Observed: {obs_distance:.3f}')
        hist_axes[i].set_xlim(xmin, xmax)
    
    hist_axes[0].set_ylabel('Frequency', fontsize=11)
    hist_axes[-1].set_xlabel('Fréchet Distance', fontsize=11)#, x=-0.15)
    hist_axes[0].set_title('Permutation Distribution', fontsize=12, fontweight='bold')#, x=0.9)
    
    # Ensure y-axis is hidden for second histogram
    if len(hist_axes) > 1:
        hist_axes[1].spines['left'].set_visible(False)
        hist_axes[1].set_yticks([])
        hist_axes[1].tick_params(left=False, labelleft=False)
    
    # CDF
    sorted_dists = np.sort(perm_distances)
    cdf = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
    ax2.plot(sorted_dists, cdf, 'b-', linewidth=2)
    ax2.axvline(obs_distance, color='red', linestyle='--', linewidth=2.5,
               label=f'Observed (p={pval:.4f})')
    ax2.set_xlabel('Fréchet Distance', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'frechet_permutation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {Path(save_dir) / 'frechet_permutation_summary.png'}")
    
    return obs_distance, pval


def dummy_example():
    # Generate example data
    np.random.seed(42)
    
    # Two different distributions
    X = np.random.randn(1000, 10) * 1.0 + 0.0
    Y = np.random.randn(1000, 10) * 1.0 + 0.0  # Shifted mean
    print("Running Fréchet permutation test with animation...")
    
    # Create animated visualization
    
    obs, pval = frechet_permutation_test_animated(
        X, Y, 
        n_perms=30, 
        seed=0,
        save_dir='./',
        fps=15,
        n_bins=25
    )
    
    # Create static summary
    create_static_summary_plot(
        X, Y,
        n_perms=30,
        seed=0,
        save_dir='./',
        n_bins=25
    )

def real_example():
    pass
    obs, pval = frechet_permutation_test_animated(
        X, Y,
        n_perms=30,
        seed=0,
        save_dir='./',
        fps=15,
        n_bins=25
    )
    create_static_summary_plot(
        X, Y,
        n_perms=30,
        seed=0,
        save_dir='./',
        n_bins=25
    )

if __name__ == "__main__":
    dummy_example()