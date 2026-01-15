import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import time
import os

"""
Hydrodynamic Quantum Analogs: 2D Pilot-Wave Simulation
======================================================
Emergence of Born's Rule and phase current from local dynamics.

This CPU implementation is slow, it serves as an algorithmic reference. 
It work well for simulating one particle (or a few),
but for better statistical convergence (high N_runs), 
you may try to use the GPU/Taichi version.


Author:      Revoire Christian 
Affiliation: Independent Researcher
Date:        Janvier 2026
License:     MIT
"""

# ===============================
# CONFIG IMPORT
# ===============================
try:
    from src.config import Born_2D_Config
except ImportError:
    from config import Born_2D_Config
CFG = Born_2D_Config()

# ===============================
# NUMERICAL CORE (NUMBA)
# ===============================

@njit(parallel=True, fastmath=True)
def evolve_field_2d(psi, psi_new, lap_buffer, x_p, y_p, 
                    dt, dx, dy, omega, gamma, d_psi, 
                    emit_amp, sigma, x_min, y_min, Nx, Ny):
    """
    Evolves the complex guiding field ψ(x,t) for one time step.

    Contributions:
    - Diffusion (Laplacian)
    - Oscillation (imaginary term)
    - Linear Damping
    - Particle emission (localized Gaussian source)
    """
    # 1) Compute 2D Laplacian (parallel loops)
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            lap_buffer[i,j] = (
                (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] - 4*psi[i,j]) 
                / dx**2
            )

    # 2) Field update 
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            psi_new[i,j] = psi[i,j] + dt * (
                (d_psi + 1j*omega) * lap_buffer[i,j] 
                - gamma * psi[i,j]
            )

    # 3) Mobile localized source 
    cutoff = int(5 * sigma / dx)
    ix = int(round((x_p - x_min) / dx))
    iy = int(round((y_p - y_min) / dy))

    i_min = max(1, ix - cutoff)
    i_max = min(Nx-1, ix + cutoff + 1)
    j_min = max(1, iy - cutoff)
    j_max = min(Ny-1, iy + cutoff + 1)

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            xi = x_min + i * dx
            yj = y_min + j * dy
            dist2 = (xi - x_p)**2 + (yj - y_p)**2
            psi_new[i,j] += emit_amp * np.exp(-0.5 * dist2 / sigma**2) * dt

    return psi_new


@njit(fastmath=True)
def get_guidance_2d(psi, x_p, y_p, x_min, y_min, dx, dy, Nx, Ny, epsilon):
    """
    Compute a robust local amplitude and phase gradient at the particle
    position using bilinear interpolation. Phase differences are
    unwrapped to handle 2π jumps safely.

    Returns:
        amp : local (interpolated) amplitude
        gx  : ∂_x phase (approx.)
        gy  : ∂_y phase (approx.)
    """
    ix = int(round((x_p - x_min) / dx))
    iy = int(round((y_p - y_min) / dy))

    # Boundary safety
    if not (1 <= ix < Nx-2 and 1 <= iy < Ny-2):
        return 0.0, 0.0, 0.0

    # Bilinear interpolation for amplitude
    fx = (x_p - (x_min + ix*dx)) / dx
    fy = (y_p - (y_min + iy*dy)) / dy

    a00 = np.abs(psi[ix, iy])
    a10 = np.abs(psi[ix+1, iy])
    a01 = np.abs(psi[ix, iy+1])
    a11 = np.abs(psi[ix+1, iy+1])
    amp = (1-fx)*(1-fy)*a00 + fx*(1-fy)*a10 + (1-fx)*fy*a01 + fx*fy*a11

    if amp < epsilon:
        return amp, 0.0, 0.0

    # Helper: robust phase difference with unwrapping
    def phase_diff(p1, p2):
        d = np.angle(p1) - np.angle(p2)
        if d > np.pi:
            d -= 2*np.pi
        elif d < -np.pi:
            d += 2*np.pi
        return d

    gx = phase_diff(psi[ix+1, iy], psi[ix-1, iy]) / (2*dx)
    gy = phase_diff(psi[ix, iy+1], psi[ix, iy-1]) / (2*dy)

    return amp, gx, gy


@njit(fastmath=True)
def simulate_particle_2d(x_init, y_init, N_steps, thermalization, subsample,
                         dt, dx, dy, omega, gamma, d_psi, emit_amp, sigma,
                         alpha, D_x, epsilon, x_min, x_max, y_min, y_max, 
                         Nx, Ny):
    """
    Simulate a single 2D particle coupled to its own pilot-wave field.

    Returns:
        positions_x, positions_y : sampled particle positions after thermalization
        psi_acc   : time-accumulated complex field (for guiding)
        psi2_acc  : time-accumulated intensity |ψ|² (for convergence)

    The routine uses explicit Euler updates and performs a gentle
    renormalization/clipping for numerical stability.
    """
    # Initialization
    psi = np.zeros((Nx, Ny), dtype=np.complex64)
    psi_new = np.zeros_like(psi)
    lap_buffer = np.zeros_like(psi)

    x_p, y_p = x_init, y_init

    # Storage
    n_samples = (N_steps - thermalization) // subsample
    positions_x = np.zeros(n_samples, dtype=np.float32)
    positions_y = np.zeros(n_samples, dtype=np.float32)
    psi_acc = np.zeros((Nx, Ny), dtype=np.complex64)
    psi2_acc = np.zeros((Nx, Ny), dtype=np.float32)

    sample_idx = 0

    # Time loop
    for t in range(N_steps):
        # Advance the field by one time-step
        psi_new = evolve_field_2d(psi, psi_new, lap_buffer, x_p, y_p,
                                  dt, dx, dy, omega, gamma, d_psi,
                                  emit_amp, sigma, x_min, y_min, Nx, Ny)
        psi[:, :] = psi_new[:, :]

        # Soft normalization (stability)
        norm2 = 0.0
        for i in range(Nx):
            for j in range(Ny):
                norm2 += psi[i,j].real**2 + psi[i,j].imag**2
        norm = np.sqrt(norm2 * dx * dy)

        if norm > 10.0:
            factor = 10.0 / norm
            for i in range(Nx):
                for j in range(Ny):
                    psi[i,j] *= factor

        # Guidance: local amplitude and phase gradient
        amp, gx, gy = get_guidance_2d(psi, x_p, y_p, x_min, y_min,
                                      dx, dy, Nx, Ny, epsilon)

        amp2 = amp**2
        if amp2 > epsilon**2:
            weight = amp2 / (amp2 + epsilon**2)
            drift_x = alpha * weight * gx
            drift_y = alpha * weight * gy

            # Clip for stability
            if drift_x > 10.0: drift_x = 10.0
            elif drift_x < -10.0: drift_x = -10.0
            if drift_y > 10.0: drift_y = 10.0
            elif drift_y < -10.0: drift_y = -10.0
        else:
            drift_x, drift_y = 0.0, 0.0

        # Overdamped Langevin step
        noise_x = np.sqrt(2 * D_x * dt) * np.random.randn()
        noise_y = np.sqrt(2 * D_x * dt) * np.random.randn()

        x_p += drift_x * dt + noise_x
        y_p += drift_y * dt + noise_y

        # Reflecting particle boundaries
        if x_p < x_min + 2*dx:
            x_p = x_min + 2*dx
        elif x_p > x_max - 2*dx:
            x_p = x_max - 2*dx

        if y_p < y_min + 2*dy:
            y_p = y_min + 2*dy
        elif y_p > y_max - 2*dy:
            y_p = y_max - 2*dy

        # Accumulate samples and fields after thermalization
        if t >= thermalization:
            if (t - thermalization) % subsample == 0:
                positions_x[sample_idx] = x_p
                positions_y[sample_idx] = y_p
                sample_idx += 1

            # Accumulate field and intensity
            for i in range(Nx):
                for j in range(Ny):
                    psi_acc[i,j] += psi[i,j]
                    psi2_acc[i,j] += psi[i,j].real**2 + psi[i,j].imag**2

    return positions_x, positions_y, psi_acc, psi2_acc


# ===============================
# PARALLEL WORKER
# ===============================

def worker_particle_2d(seed, particle_id, x_space, y_space):
    """
    Worker executed on a separate CPU core. Simulates a single 2D particle
    and returns light-weight statistics (2D histogram, accumulated fields,
    and optionally a short trajectory for visualization).
    """
    np.random.seed(seed)

    # Initial radial distribution for starting positions
    r0 = np.random.uniform(5, 10)
    theta0 = np.random.uniform(0, 2*np.pi)
    x_init = r0 * np.cos(theta0)
    y_init = r0 * np.sin(theta0)

    # Run single-particle simulation
    pos_x, pos_y, psi_acc, psi2_acc = simulate_particle_2d(
        x_init, y_init, CFG.N_steps, CFG.thermalization, CFG.SUBSAMPLE,
        CFG.dt, CFG.dx, CFG.dy, CFG.omega, CFG.gamma, CFG.D_psi, CFG.emit_amp, CFG.sigma_emit_scaled,
        CFG.alpha, CFG.D_x, CFG.epsilon, CFG.x_min, CFG.x_max, CFG.y_min, CFG.y_max,
        CFG.Nx, CFG.Ny
    )

    # 2D histogram of visited positions
    hist, _, _ = np.histogram2d(pos_x, pos_y,
                                bins=[len(x_space), len(y_space)],
                                range=[[CFG.x_min, CFG.x_max], [CFG.y_min, CFG.y_max]])

    # Save a few trajectories
    save_traj = (particle_id < 5)  # keep first 5 particle trajectories
    if save_traj:
        traj = np.stack([pos_x[::10], pos_y[::10]], axis=1)
    else:
        traj = None

    return {
        'histogram': hist.astype(np.float32),
        'psi_acc': psi_acc,
        'psi2_acc': psi2_acc,
        'n_samples': len(pos_x),
        'trajectory': traj
    }


# ===============================
# MAIN SIMULATION 
# ===============================

def run_born_simulation_2d():
    """
    Orchestrate an ensemble of independent 2D single-particle simulations.

    The function builds the spatial grids,  launches
    joblib workers, aggregates ensemble-averaged fields and histograms,
    and returns processed quantities ready for analysis and plotting.
    """
    # Detect number of CPU cores to use
    n_cores = CFG.N_CORES if CFG.N_CORES > 0 else max(1, mp.cpu_count() + CFG.N_CORES)

    # Spatial grids
    x_space = np.linspace(CFG.x_min, CFG.x_max, CFG.Nx)
    y_space = np.linspace(CFG.y_min, CFG.y_max, CFG.Ny)
    X, Y = np.meshgrid(x_space, y_space, indexing='ij')

    # Memory estimate
    n_samples_per_particle = (CFG.N_steps - CFG.thermalization) // CFG.SUBSAMPLE
    memory_per_particle_mb = (
        (n_samples_per_particle * 2 * 4 +  # positions (float32)
         CFG.Nx * CFG.Ny * 8 +  # psi_acc (complex64)
         CFG.Nx * CFG.Ny * 4)   # psi2_acc (float32)
        / (1024**2)
    )
    total_memory_mb = memory_per_particle_mb * CFG.N_runs / n_cores

    # CFL-like stability metric
    cfl = CFG.omega * CFG.dt / CFG.dx**2

    print("="*70)
    print("SIMULATION 2D BORN RULE")
    print("="*70)
    print(f"Configuration :")
    print(f"  - Grid : {CFG.Nx}×{CFG.Ny} ({CFG.Nx*CFG.Ny:,} points)")
    print(f"  - CPU cores : {n_cores}/{mp.cpu_count()}")
    print(f"  - Particles : {CFG.N_runs}")
    print(f"  - Steps/particle : {CFG.N_steps:,}")
    print(f"  - Subsampling : 1/{CFG.SUBSAMPLE}")
    print(f"  - Estimated memory : {total_memory_mb:.1f} MB")
    print(f"\nNumerical stability :")
    print(f"  - CFL = {cfl:.4f} (should be < 0.5)")
    print(f"  - dx = {CFG.dx:.3f}, dy = {CFG.dy:.3f}, dt = {CFG.dt}")
    print(f"\nPhysical params :")
    print(f"  ω={CFG.omega}, γ={CFG.gamma}, D_ψ={CFG.D_psi}")
    print(f"  α={CFG.alpha}, D_x={CFG.D_x} {'✓ ACTIVE' if CFG.D_x > 0 else '⚠️ INACTIVE'}")
    print(f"  emit_amp={CFG.emit_amp}, σ={CFG.sigma_emit_scaled :.2f}")
    print("="*70)

    if cfl >= 0.5:
        print("⚠️  WARNING : CFL >= 0.5, numerical instability possible!")

    start_time = time.time()

    # Launch parallel workers
    print("\n🚀 Launching parallel simulations...\n")

    results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
        delayed(worker_particle_2d)(
            seed=42 + p*1000,
            particle_id=p,
            x_space=x_space,
            y_space=y_space
        ) for p in tqdm(range(CFG.N_runs), desc="Particles")
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed/60:.2f} min")
    print(f"  Speed : {CFG.N_runs * CFG.N_steps / elapsed / 1e6:.2f}M steps/sec")

    # Aggregate ensemble statistics
    print("\n📊 Aggregating statistics...")

    rho = np.zeros((CFG.Nx, CFG.Ny))
    psi_acc = np.zeros((CFG.Nx, CFG.Ny), dtype=np.complex128)
    psi2_acc = np.zeros((CFG.Nx, CFG.Ny))
    total_samples = 0
    trajectories = []

    for res in tqdm(results, desc="Merge"):
        rho += res['histogram']
        psi_acc += res['psi_acc']
        psi2_acc += res['psi2_acc']
        total_samples += res['n_samples']

        if res['trajectory'] is not None:
            trajectories.append(res['trajectory'])

    # Normalize empirical density
    rho_smooth = gaussian_filter(rho, sigma=1.5)
    rho_smooth /= np.trapz(np.trapz(rho_smooth, x_space, axis=0), y_space)

    psi_eff = psi_acc / total_samples
    born = psi2_acc / total_samples
    born /= np.trapz(np.trapz(born, x_space, axis=0), y_space)

    return X, Y, x_space, y_space, rho_smooth, born, psi_eff, trajectories


# ===============================
# ANALYSIS
# ===============================

def analyze_results(X, Y, x_space, y_space, rho, born, psi_eff):
    """
    Compute quantitative diagnostics:
    - spatial correlation between empirical density and |ψ|²
    - L1 error
    - probability current J = Im(ψ* ∇ψ) and average angular momentum <Lz>

    Returns: (corr, error_L1, J_x, J_y)
    """
    # Correlation using masked points to avoid divisions by zero
    mask = (rho > 1e-6) & (born > 1e-6)
    if np.sum(mask) > 100:
        corr = np.corrcoef(rho[mask], born[mask])[0,1]
    else:
        corr = 0.0

    # L1 error
    error_L1 = 0.5 * np.trapz(np.trapz(np.abs(rho - born), x_space, axis=0), y_space)

    # Probability current (vectorial polarization)
    grad_y, grad_x = np.gradient(psi_eff)
    J_x = np.imag(np.conj(psi_eff) * grad_x)
    J_y = np.imag(np.conj(psi_eff) * grad_y)

    # Mean angular momentum density and total expectation value
    Lz_field = X * J_y - Y * J_x
    Lz_mean = np.trapz(np.trapz(Lz_field * born, x_space, axis=0), y_space)

    print("\n" + "="*70)
    print("CONVERGENCE TOWARD |ψ|²")
    print("="*70)
    print(f"ρ vs |ψ|² correlation : {corr:.4f}")
    print(f"L¹ error : {error_L1:.6f}")
    print(f"<Lz> (quantum-like) : {Lz_mean:.3f}")

    return corr, error_L1, J_x, J_y


# ===============================
# VISUALIZATION
# ===============================

def plot_results_2d_improved(X, Y, rho, born, psi_eff, J_x, J_y, trajectories, corr, error_L1):
    """
    Create a 2×3 figure showing:
      1) empirical density ρ(x,y)
      2) Born density |ψ|²
      3) residuals (|ψ|² - ρ)
      4) probability current J (quiver)
      5) phase (in units of π)
      6) sample trajectories
    """
    fig = plt.figure(figsize=(18, 10))

    x_space = X[:, 0]
    y_space = Y[0, :]
    dx = x_space[1] - x_space[0]

    # 1) Particle density (converted to parts-per-ten-thousand for display)
    ax1 = plt.subplot(2, 3, 1)
    rho_percent = rho * 10000  # display units
    im1 = ax1.contourf(X, Y, rho_percent, levels=30, cmap='viridis')
    ax1.set_title(r'ρ(x,y) - Particle density', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Probability (‱ per u.a.^2)')
    ax1.set_aspect('equal')
    ax1.text(0.05, 0.95, r'∫ ρ dA = 1', transform=ax1.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2) Born density
    ax2 = plt.subplot(2, 3, 2)
    born_percent = born * 10000
    im2 = ax2.contourf(X, Y, born_percent, levels=30, cmap='viridis')
    ax2.set_title(r'$|\psi|^2$ - Born rule', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Probability (‱ per u.a.^2)')
    ax2.set_aspect('equal')
    ax2.text(0.05, 0.95, r'∫ |ψ|^2 dA = 1', transform=ax2.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3) Residuals
    ax3 = plt.subplot(2, 3, 3)
    diff = rho - born
    vmax = max(np.max(np.abs(diff)), 1e-6)
    im3 = ax3.contourf(X, Y, diff*10000, levels=30, cmap='RdBu_r', vmin=-vmax*10000, vmax=vmax*10000)
    if vmax > 1e-4:
        n_contours = 7
        contour_levels = np.linspace(-vmax*0.8, vmax*0.8, n_contours) * 100
        cs = ax3.contour(X, Y, diff * 100, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.5)
        ax3.clabel(cs, inline=True, fontsize=7, fmt='%.2e')
    ax3.set_title(r'$|\psi|^2 - \rho(x,y)$ - Residuals (1e-5)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3)
    ax3.set_aspect('equal')

    # 4) Phase current (quiver)
    ax4 = plt.subplot(2, 3, 4)
    step = 2
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    Jx_sub = J_x[::step, ::step]
    Jy_sub = J_y[::step, ::step]
    density_bg = np.abs(psi_eff)**2
    ax4.contourf(X, Y, density_bg * 100, levels=20, cmap='gray', alpha=0.3)
    J_norm = np.sqrt(J_x**2 + J_y**2)
    max_J = np.max(J_norm) if np.max(J_norm) > 0 else 1.0
    Q = ax4.quiver(X_sub, Y_sub, Jx_sub, Jy_sub, color='red', alpha=0.8, scale=max_J*20, width=0.003, headwidth=4)
    ax4.set_title(r'Current $abla J = \mathrm{Im}(\psi^* \nabla \psi)$', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_aspect('equal')
    ax4.text(0.05, 0.95, r'$\vec{J} = \mathrm{Im}(\psi^* \nabla \psi)$', transform=ax4.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5) Phase 
    ax5 = plt.subplot(2, 3, 5)
    phase = np.angle(psi_eff)
    phase_pi = phase / np.pi
    from scipy.ndimage import gaussian_filter as _gf
    phase_pi_smooth = _gf(phase_pi, sigma=2.0)
    im5 = ax5.contourf(X, Y, phase_pi_smooth, levels=np.linspace(-1, 1, 31), cmap='twilight')
    ax5.set_title('Wave phase', fontsize=13, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    cbar5 = plt.colorbar(im5, ax=ax5, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar5.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    cbar5.set_label('Phase (rad)')
    ax5.set_aspect('equal')

    # 6) Trajectories
    ax6 = plt.subplot(2, 3, 6)
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    for traj in trajectories:
        if len(traj) > 0:
            ax6.plot(traj[:,0], traj[:,1], alpha=0.7, lw=1.2)
    ax6.contour(X, Y, rho, levels=8, alpha=0.4, colors='gray')
    ax6.set_title(f'Trajectories (N={len(trajectories)})', fontsize=13, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_aspect('equal')
    ax6.grid(alpha=0.3)

    # Super title
    fig.suptitle(f"Emergence of Born's rule 2D (N={CFG.N_runs}, Steps={CFG.N_steps})\n" + \
                 f"ρ vs |ψ|² corr: {corr:.4f} | L¹: {error_L1:.5f}", fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    base = f"Born_2D_N{CFG.N_runs}"
    i = 1
    while os.path.exists(f"{base}_V{i}.png"):
        i += 1
    filename = f"{base}_V{i}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Plot saved: {filename}")

    plt.show()


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    # Run the main 2D Born simulation and visualize results
    X, Y, x_space, y_space, rho, born, psi_eff, trajectories = run_born_simulation_2d()

    # Analysis
    corr, error_L1, J_x, J_y = analyze_results(X, Y, x_space, y_space, rho, born, psi_eff)

    # Plot
    plot_results_2d_improved(X, Y, rho, born, psi_eff, J_x, J_y, trajectories, corr, error_L1)

    print("\n" + "="*70)
    print("✓ 2D SIMULATION COMPLETED")
    print("="*70)
