import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp

"""
Hydrodynamic Quantum Analogs: 1D Pilot-Wave Simulation
======================================================
Emergence of Pauli Exclusion Principle from Local Dynamics. 


Author:      Revoire Christian 
Affiliation: Independent Researcher
Date:        Janvier 2026
License:     MIT
"""


# ===============================
# CONFIGURATION IMPORT
# ===============================
try:
    from src.config import Pauli_Config
except ImportError:
    from config import Pauli_Config
CFG = Pauli_Config()

"""
Parameters are loaded from config.py.

To change physics or simulation settings, edit 'src/config.py' 
"""

# ===============================
# PHYSICAL CORE (Numba)
# ===============================

@njit(fastmath=True)
def get_drift(psi_val, psi_prev, psi_next, dx, epsilon, alpha):
    """
    Computes the local drift velocity from the phase gradient
    of the guiding field.

    This is the core guidance law of the model.
    """
  
    amp2 = np.abs(psi_val)**2
    # Negligible field amplitude → no reliable phase information
    if amp2 < epsilon**2:
        return 0.0
      
    # Prevents ill-defined phase when |ψ| ≈ 0
    if not np.isfinite(psi_prev) or not np.isfinite(psi_next) or not np.isfinite(psi_val):
        return 0.0

    amp_prev = np.abs(psi_prev)
    amp_next = np.abs(psi_next)
  
    # Avoid phase jumps across near-zero amplitudes
    if amp_prev < epsilon or amp_next < epsilon:
        return 0.0
      
    # Discrete phase gradient
    dph = np.angle(psi_next) - np.angle(psi_prev)

    # Phase unwrapping
    if dph > np.pi: dph -= 2*np.pi
    elif dph < -np.pi: dph += 2*np.pi
    
    if not np.isfinite(dph):
        return 0.0
    
    grad_phase = dph / (2 * dx)
    weight = amp2 / (amp2 + epsilon**2)

    # Guidance equation
    drift = alpha * weight * grad_phase
    
    if not np.isfinite(drift):
        return 0.0
      
    # Numerical stability: limit extreme velocities
    if drift > 10.0: drift = 10.0
    elif drift < -10.0: drift = -10.0
    
    return drift

@njit(fastmath=True)
def evolve_field(psi, x_particle, dt, dx, c, D_psi, omega, gamma, 
                emit_amp, sigma_emit, x_min, Nx):
    """
    Evolves the complex guiding field ψ(x,t) for one time step.

    Contributions:
    - Diffusion (Laplacian)
    - Oscillation (imaginary term)
    - Linear Damping
    - Particle emission (localized Gaussian source)
    """
                  
    # Discrete Laplacian
    lap = np.zeros_like(psi)
    for i in range(1, Nx-1):
        lap[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2

    # Field evolution: diffusion + oscillation + damping
    psi_new = psi + dt * c * ((D_psi + 1j * omega) * lap - gamma * psi)

    # Localized particle emission
    cutoff = 6.0 * sigma_emit
    cutoff_idx = int(cutoff / dx)
    idx = int(round((x_particle - x_min) / dx))
    
    i_start = max(0, idx - cutoff_idx)
    i_end = min(Nx, idx + cutoff_idx + 1)
    
    for i in range(i_start, i_end):
        xi = x_min + i * dx
        dist2 = (xi - x_particle)**2
        # Gaussian emission centered on the particle
        psi_new[i] += emit_amp * np.exp(-0.5 * dist2 / sigma_emit**2) * dt
    
    return psi_new

@njit(fastmath=True)
def simulate_interaction(x1_init, x2_init, N_steps, dt, dx, c, D_psi, omega, gamma,
                        emit_amp, sigma_emit, alpha, D_x, epsilon, 
                        x_min, x_max, Nx, coupling, thermalization):

    """
    This function implements the full two-particle model:
    • Each particle emits its own complex guiding field
    • Both particles are guided by the phase gradient of a
      combined effective field (ψ₁ ± ψ₂)
    • No antisymmetrization or exclusion rule is imposed
    • Stochastic diffusion ensures ergodicity

    After a thermalization period, ensemble-averaged fields
    and Born densities are accumulated.
    """

    # Individual guiding fields for each particle
    psi1 = np.zeros(Nx, dtype=np.complex128)
    psi2 = np.zeros(Nx, dtype=np.complex128)
    
    x1, x2 = x1_init, x2_init
    traj1 = np.zeros(N_steps)
    traj2 = np.zeros(N_steps)

    # Field and density accumulation after thermalization
    psi1_acc = np.zeros(Nx, dtype=np.complex128)
    psi2_acc = np.zeros(Nx, dtype=np.complex128)
    psi_sum_acc = np.zeros(Nx, dtype=np.complex128)

    born1_acc = np.zeros(Nx, dtype=np.float64)
    born2_acc = np.zeros(Nx, dtype=np.float64)
    born_sum_acc = np.zeros(Nx, dtype=np.float64)

    n_acc = 0

    # Time evolution loop
    for t in range(N_steps):
        psi1 = evolve_field(psi1, x1, dt, dx, c, D_psi, omega, gamma,
                           emit_amp, sigma_emit, x_min, Nx)
        psi2 = evolve_field(psi2, x2, dt, dx, c, D_psi, omega, gamma,
                           emit_amp, sigma_emit, x_min, Nx)

        # Effective guiding field.
        # Note: no symmetry constraint is imposed on trajectories.
        # Any exclusion effect must emerge dynamically.
        if coupling == 1:
            psi_guide = psi1 + psi2
        else:
            psi_guide = psi1 - psi2    # equivalent to a phase shift of π : ψ₁ - ψ₂ = ψ₁(0) + ψ₂(π)
        
        idx1 = int(round((x1 - x_min) / dx))
        idx2 = int(round((x2 - x_min) / dx))
        
        d1, d2 = 0.0, 0.0
        if 1 < idx1 < Nx-2:
            d1 = get_drift(psi_guide[idx1], psi_guide[idx1-1], 
                          psi_guide[idx1+1], dx, epsilon, alpha)
        if 1 < idx2 < Nx-2:
            d2 = get_drift(psi_guide[idx2], psi_guide[idx2-1],
                          psi_guide[idx2+1], dx, epsilon, alpha)
          
        # Stochastic diffusion term.
        # Ensures ergodicity and exploration of configuration space.
        noise1 = np.sqrt(2 * D_x * dt) * np.random.randn()
        noise2 = np.sqrt(2 * D_x * dt) * np.random.randn()
        
        x1 += d1 * dt + noise1
        x2 += d2 * dt + noise2
        
        if x1 < x_min + dx: x1 = x_min + dx
        elif x1 > x_max - dx: x1 = x_max - dx
        if x2 < x_min + dx: x2 = x_min + dx
        elif x2 > x_max - dx: x2 = x_max - dx
        
        traj1[t] = x1
        traj2[t] = x2
        
        # Accumulation after thermalization.
        if t >= thermalization:
            psi1_acc += psi1
            psi2_acc += psi2
            psi_sum_acc += psi_guide
            born1_acc += np.abs(psi1)**2
            born2_acc += np.abs(psi2)**2
            born_sum_acc += np.abs(psi_guide)**2
            n_acc += 1
    
    # Normalization
    if n_acc > 0:
        psi1_acc /= n_acc
        psi2_acc /= n_acc
        psi_sum_acc /= n_acc
        born1_acc /= n_acc
        born2_acc /= n_acc
        born_sum_acc /= n_acc
    
    return traj1, traj2, psi1_acc, psi2_acc, psi_sum_acc, born1_acc, born2_acc, born_sum_acc

@njit(fastmath=True)
def simulate_solo(x_init, N_steps, dt, dx, c, D_psi, omega, gamma,
                 emit_amp, sigma_emit, alpha, D_x, epsilon,
                 x_min, x_max, Nx, thermalization):

    """
    Simulate the dynamics of a single particle interacting
    only with its own pilot-wave field.

    This function serves as a control experiment to verify
    the dynamical convergence toward the Born rule
    
    I generates 'ghost trajectories' serving as 
    an uncorrelated reference for pair statistics.
    No interaction or coupling is present.
    """


    psi = np.zeros(Nx, dtype=np.complex128)
    x = x_init
    traj = np.zeros(N_steps)
                   
    # Field and density accumulation after thermalization
    psi_accumulated = np.zeros(Nx, dtype=np.complex128)
    born_accumulated = np.zeros(Nx, dtype=np.float64)
    n_acc = 0
    
    for t in range(N_steps):
        psi = evolve_field(psi, x, dt, dx, c, D_psi, omega, gamma,
                          emit_amp, sigma_emit, x_min, Nx)
        
        idx = int(round((x - x_min) / dx))
        d = 0.0
        if 1 < idx < Nx-2:
            d = get_drift(psi[idx], psi[idx-1], psi[idx+1], dx, epsilon, alpha)
        
        noise = np.sqrt(2 * D_x * dt) * np.random.randn()
        x += d * dt + noise
        
        if x < x_min + dx: x = x_min + dx
        elif x > x_max - dx: x = x_max - dx
        
        traj[t] = x
        
        # Accumulation after thermalization.
        if t >= thermalization:
            psi_accumulated += psi
            born_accumulated += np.abs(psi)**2
            n_acc += 1
    
    # Normalization
    if n_acc > 0:
        psi_accumulated /= n_acc
        born_accumulated /= n_acc
    
    return traj, psi_accumulated, born_accumulated

# ===============================
# FONCTION g(r)
# ===============================
"""
Pair correlation function g(r).

Defined as:
    g(r) = ⟨ ρ(x₁, x₂) ⟩ / (ρ₁(x₁) ρ₂(x₂))     
         = P_real(r) / P_uncorrelated(r)

with r = |x₁ − x₂|.

The denominator is obtained from "ghost" trajectories,
i.e. two independent single-particle simulations.

For fermions:
-------------
• g(0) → 0   (Pauli exclusion)
• g(r) shows a correlation hole at short distances
• Joint configurations near x₁ = x₂ are suppressed
"""

def compute_pair_correlation(hist_real, hist_ghost, r_centers):
    hist_real_norm = hist_real / (np.sum(hist_real) * (r_centers[1] - r_centers[0]))
    hist_ghost_norm = hist_ghost / (np.sum(hist_ghost) * (r_centers[1] - r_centers[0]))
    
    g_r = np.divide(hist_real_norm, hist_ghost_norm, 
                    out=np.ones_like(hist_real_norm), 
                    where=(hist_ghost_norm > 1e-10))
    
    return r_centers, g_r, hist_real_norm, hist_ghost_norm

# ===============================
# WORKER PARALLÈLE
# ===============================

def worker_particle(seed, particle_id, x_space, coupling_code, side, 
                    start_area_p1, start_area_p2, bins_1d, bins_2d, bin_edges_dist):
    """
    Executes one independent stochastic realization of the
    two-particle pilot-wave dynamics.

    Individual trajectories do not exhibit explicit exclusion.
    But The Pauli exclusion principle statistically emerges after
    ensemble averaging.

    Each worker produces:
    • One interacting (real) trajectory pair
    • Two independent (ghost) trajectories
    • Time-averaged fields and densities
    • Pairwise distance statistics
    """
  
    np.random.seed(seed)
    
    if side == "rand":
        # Random left/right assignment (50% inversion)
        if np.random.rand() < 0.5:
            area_p1 = (-15.0, -5.0)  # P1 gauche
            area_p2 = (5.0, 15.0)    # P2 droite
        else:
            area_p1 = (5.0, 15.0)    # P1 droite
            area_p2 = (-15.0, -5.0)  # P2 gauche
    else:
        # Uses globally defined starting regions
        area_p1 = start_area_p1
        area_p2 = start_area_p2
    
    x1_init = np.random.uniform(area_p1[0], area_p1[1])
    x2_init = np.random.uniform(area_p2[0], area_p2[1])
    
    # 1. INTERACTION 
    t1, t2, psi1_acc, psi2_acc, psi_sum_acc, born1_acc, born2_acc, born_sum_acc = simulate_interaction(
        x1_init, x2_init, CFG.N_steps, CFG.dt, CFG.dx, CFG.c, CFG.D_psi, CFG.omega, CFG.gamma,
        CFG.emit_amp, CFG.sigma_emit_scaled, CFG.alpha, CFG.D_x, CFG.epsilon,
        CFG.x_min, CFG.x_max, CFG.Nx, coupling_code, CFG.thermalization
    )
    
    # 2. SOLO P1 
    ts1, phi1_acc, born1_accumulated = simulate_solo(
        x1_init, CFG.N_steps, CFG.dt, CFG.dx, CFG.c, CFG.D_psi, CFG.omega, CFG.gamma,
        CFG.emit_amp, CFG.sigma_emit_scaled, CFG.alpha, CFG.D_x, CFG.epsilon,
        CFG.x_min, CFG.x_max, CFG.Nx, CFG.thermalization
    )
    
    # 3. SOLO P2
    ts2, phi2_acc, born2_accumulated = simulate_solo(
        x2_init, CFG.N_steps, CFG.dt, CFG.dx, CFG.c, CFG.D_psi, CFG.omega, CFG.gamma,
        CFG.emit_amp, CFG.sigma_emit_scaled, CFG.alpha, CFG.D_x, CFG.epsilon,
        CFG.x_min, CFG.x_max, CFG.Nx, CFG.thermalization
    )
    
    # Post-thermalization Analyse
    t1_post = t1[CFG.thermalization:]
    t2_post = t2[CFG.thermalization:]
    ts1_post = ts1[CFG.thermalization:]
    ts2_post = ts2[CFG.thermalization:]
    
    # Position for Histograms
    hist_p1_real, _ = np.histogram(t1_post, bins=len(x_space), range=(CFG.x_min, CFG.x_max))
    hist_p2_real, _ = np.histogram(t2_post, bins=len(x_space), range=(CFG.x_min, CFG.x_max))
    
    # Distances
    d_real = np.abs(t1_post - t2_post)
    d_ghost = np.abs(ts1_post - ts2_post)
    hist_dist_real, _ = np.histogram(d_real, bins=bin_edges_dist)
    hist_dist_ghost, _ = np.histogram(d_ghost, bins=bin_edges_dist)
                      
    # Position for Heatmap 
    hist_2d, _, _ = np.histogram2d(
        t1_post[::1],     # Subsampling possible
        t2_post[::1],
        bins=bins_2d,
        range=[[CFG.x_min, CFG.x_max], [CFG.x_min, CFG.x_max]]
    )
    
    return {
        'hist_p1': hist_p1_real.astype(np.float32),
        'hist_p2': hist_p2_real.astype(np.float32),
        'hist_dist_real': hist_dist_real.astype(np.float32),
        'hist_dist_ghost': hist_dist_ghost.astype(np.float32),
        'hist_2d': hist_2d.astype(np.float32),
        'psi1_acc': psi1_acc,
        'psi2_acc': psi2_acc,
        'psi_sum_acc': psi_sum_acc,
        'phi1_acc': phi1_acc,
        'phi2_acc': phi2_acc,
        'born1_acc': born1_acc,
        'born2_acc': born2_acc,
        'born_sum_acc': born_sum_acc,
        'born1_accumulated': born1_accumulated,
        'born2_accumulated': born2_accumulated
    }

# ===============================
# MAIN SIMULATION
# ===============================

"""
Main orchestration routine for the Pauli exclusion simulation.

The goal is to test whether Pauli-like exclusion emerges dynamically
from local pilot-wave interactions, without imposing quantum statistics.
"""

def run_pauli_simulation():
    # Determine the number of CPU cores to use.
    n_cores = CFG.N_CORES if CFG.N_CORES > 0 else max(1, mp.cpu_count() + CFG.N_CORES)
    
    print("="*70)
    print("PAULI EXCLUSION — EMERGENT DYNAMICS")
    print("="*70)
    print(f"Configuration:")
    print(f"  - CPU Cores: {n_cores}/{mp.cpu_count()}")
    print(f"  - Particles: {CFG.N_runs} pairs")
    print(f"  - Steps/Pair: {CFG.N_steps}")
    print(f"\nPhysics :")
    print(f"  γ={CFG.gamma}, D_ψ={CFG.D_psi}, ω={CFG.omega}, α={CFG.alpha}, D_x={CFG.D_x}")
    print(f"  Coupling: {CFG.coupling_type}")
    print("="*70)

    # Coupling mode
    coupling_code = 1 if CFG.coupling_type == "sum" else 0
    
    # Space/Grid step
    x_space = np.linspace(CFG.x_min, CFG.x_max, CFG.Nx)
    bins_1d = CFG.Nx
    bins_2d = CFG.Nx
    L = CFG.x_max - CFG.x_min
    bins_dist = 150
    bin_edges_dist = np.linspace(0, L, bins_dist+1)
  
    start_time = time.time()
    
    # ========================================
    # PARALLEL EXECUTION
    # ========================================

    # Histograms → empirical particle densities
    hist_p1_total = np.zeros(bins_1d, dtype=np.float64)
    hist_p2_total = np.zeros(bins_1d, dtype=np.float64)
    hist_dist_real_total = np.zeros(bins_dist, dtype=np.float64)
    hist_dist_ghost_total = np.zeros(bins_dist, dtype=np.float64)
    hist_2d_total = np.zeros((bins_2d, bins_2d), dtype=np.float64)
    # ψ fields → time-averaged guiding fields
    psi1_mean = np.zeros(CFG.Nx, dtype=np.complex128)
    psi2_mean = np.zeros(CFG.Nx, dtype=np.complex128)
    psi_sum_mean = np.zeros(CFG.Nx, dtype=np.complex128)
    phi1_mean = np.zeros(CFG.Nx, dtype=np.complex128)
    phi2_mean = np.zeros(CFG.Nx, dtype=np.complex128)
    # |ψ|² → Born-rule reference densities
    born1_mean = np.zeros(CFG.Nx, dtype=np.float64)
    born2_mean = np.zeros(CFG.Nx, dtype=np.float64)
    born_sum_mean = np.zeros(CFG.Nx, dtype=np.float64)
    born1_accumulated_mean = np.zeros(CFG.Nx, dtype=np.float64)
    born2_accumulated_mean = np.zeros(CFG.Nx, dtype=np.float64)
  
    # Each worker runs a fully independent stochastic realization.
    print("\n🚀 Starting parallel simulation...\n")
    
    results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
        delayed(worker_particle)(
            seed=42 + p*1000,
            particle_id=p,
            x_space=x_space,
            coupling_code=coupling_code,
            side=CFG.SIDE,
            start_area_p1=CFG.start_area_p1,
            start_area_p2=CFG.start_area_p2,
            bins_1d=bins_1d,
            bins_2d=bins_2d,
            bin_edges_dist=bin_edges_dist
        ) for p in tqdm(range(CFG.N_runs), desc="Pairs")
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed/60:.2f} min")
    
    # ========================================
    # AGGREGATION OF ENSEMBLE STATISTICS
    # ========================================
    print("\n📊 Aggregating statistics...")
     
    for res in tqdm(results, desc="Fusion"):
        hist_p1_total += res['hist_p1']
        hist_p2_total += res['hist_p2']
        hist_dist_real_total += res['hist_dist_real']
        hist_dist_ghost_total += res['hist_dist_ghost']
        hist_2d_total += res['hist_2d']
        psi1_mean += res['psi1_acc']
        psi2_mean += res['psi2_acc']
        psi_sum_mean += res['psi_sum_acc']
        phi1_mean += res['phi1_acc']
        phi2_mean += res['phi2_acc']
        born1_mean += res['born1_acc']
        born2_mean += res['born2_acc']
        born_sum_mean += res['born_sum_acc']
        born1_accumulated_mean += res['born1_accumulated']
        born2_accumulated_mean += res['born2_accumulated']
    
    # Normalization
    psi1_mean /= CFG.N_runs
    psi2_mean /= CFG.N_runs
    psi_sum_mean /= CFG.N_runs
    phi1_mean /= CFG.N_runs
    phi2_mean /= CFG.N_runs
    born1_mean /= CFG.N_runs
    born2_mean /= CFG.N_runs
    born_sum_mean /= CFG.N_runs
    born1_accumulated_mean /= CFG.N_runs
    born2_accumulated_mean /= CFG.N_runs
    
    # Observed particle densities from trajectory histograms
    rho_p1_obs = hist_p1_total / (np.sum(hist_p1_total) * CFG.dx)
    rho_p2_obs = hist_p2_total / (np.sum(hist_p2_total) * CFG.dx)
    rho_total_obs = rho_p1_obs + rho_p2_obs
    
    # Born densities extracted from time-averaged guiding fields
    born_p1 = born1_mean                    
    born_p1 /= np.trapz(born_p1, x_space)
    
    born_p2 = born2_mean                    
    born_p2 /= np.trapz(born_p2, x_space)
        
    integral_born = np.trapz(born_sum_mean, x_space)        
    born_sum = 2 * (born_sum_mean / integral_born)          # The total Born density is normalized to 2 (because the system contains two particles)

    r_centers = 0.5 * (bin_edges_dist[:-1] + bin_edges_dist[1:])
  
    return {
        'x_space': x_space,
        'rho_p1_obs': rho_p1_obs,
        'rho_p2_obs': rho_p2_obs,
        'rho_total_obs': rho_total_obs,
        'born_p1': born_p1,
        'born_p2': born_p2,
        'born_sum': born_sum,
        'phi1_mean': phi1_mean,
        'phi2_mean': phi2_mean,
        'hist_dist_real': hist_dist_real_total,
        'hist_dist_ghost': hist_dist_ghost_total,
        'hist_2d': hist_2d_total,
        'r_centers': r_centers,
        'bin_edges_dist': bin_edges_dist
    }

# ===============================
# THEORETICAL DENSITIES
# ===============================

def compute_theoretical_densities(x_space, phi1, phi2):
    """
    Computes theoretical one- and two-particle densities
    from standard quantum mechanics.

    These densities are NOT used in the dynamics.
    They serve only as reference distributions for comparison.
    """
  
    dx_local = x_space[1] - x_space[0]
    Nx = len(x_space)
    
    # Normalization of individual states
    norm1 = np.sqrt(np.trapz(np.abs(phi1)**2, x_space))
    norm2 = np.sqrt(np.trapz(np.abs(phi2)**2, x_space))
    
    if norm1 > 0:
        phi1 = phi1 / norm1
    if norm2 > 0:
        phi2 = phi2 / norm2
    
    print(f"   Grid calculation {Nx}×{Nx}...")

    # Full 2D configuration-space wavefunctions ψ(x₁, x₂)
    phi1_x1 = phi1[:, np.newaxis]  # (Nx, 1)
    phi2_x2 = phi2[np.newaxis, :]  # (1, Nx)
    phi1_x2 = phi1[np.newaxis, :]
    phi2_x1 = phi2[:, np.newaxis]
    
    # 2-particles wave functions
    psi_antisym = (phi1_x1 * phi2_x2 - phi1_x2 * phi2_x1) / np.sqrt(2)
    psi_sym = (phi1_x1 * phi2_x2 + phi1_x2 * phi2_x1) / np.sqrt(2)
    
    rho_2d_fermion = np.abs(psi_antisym)**2
    rho_2d_boson = np.abs(psi_sym)**2
    
    # Marginal densities obtained by integrating over the other particle.
    # Note: Pauli exclusion affects correlations, not one-particle marginals.
    rho1_fermion = np.trapz(rho_2d_fermion, x_space, axis=1)  # Integrates on x₂
    rho2_fermion = np.trapz(rho_2d_fermion, x_space, axis=0)  # Integrates on x₁
    
    rho1_boson = np.trapz(rho_2d_boson, x_space, axis=1)
    rho2_boson = np.trapz(rho_2d_boson, x_space, axis=0)
    
    # Final Normalization
    rho1_fermion /= np.trapz(rho1_fermion, x_space)
    rho2_fermion /= np.trapz(rho2_fermion, x_space)
    rho1_boson /= np.trapz(rho1_boson, x_space)
    rho2_boson /= np.trapz(rho2_boson, x_space)
    
    # Total Theoretical Densities
    rho_total_fermion = rho1_fermion + rho2_fermion
    rho_total_boson = rho1_boson + rho2_boson
    
    return {
        'rho1_fermion': rho1_fermion,
        'rho2_fermion': rho2_fermion,
        'rho1_boson': rho1_boson,
        'rho2_boson': rho2_boson,
        'rho_total_fermion': rho_total_fermion,
        'rho_total_boson': rho_total_boson,
        'rho_2d_fermion': rho_2d_fermion
    }

# ===============================
# SCHRÖDINGER COMPARISON
# ===============================

def compare_schrodinger(x_space, sigma_target):
    """
    The model simulates a freely spreading wave packet (diffusive regime).
    
    The loop below searches for the quantum time t_QM at which the width of the
    quantum packet (σ_qm) matches the model width (σ_x).
    
    This establishes the temporal scaling factor between the two dynamics: τ_stochastic / τ_Schrödinger
    
    Approximate relation: steps ∝ σ² / (ω·dt)
    """
    dx_local = x_space[1] - x_space[0]
    psi_qm = np.exp(-0.5*(x_space/2.0)**2).astype(np.complex128)
    psi_qm /= np.sqrt(np.trapz(np.abs(psi_qm)**2, x_space))

    best_error = np.inf
    best_step = 0
    best_psi = None
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        # Free evolution
        lap_qm = np.zeros_like(psi_qm)
        lap_qm[1:-1] = (psi_qm[2:] - 2*psi_qm[1:-1] + psi_qm[:-2]) / dx_local**2
        
        # Evolution
        psi_qm += CFG.dt * (1j * CFG.omega * lap_qm)
        
        # Normalization
        norm = np.sqrt(np.trapz(np.abs(psi_qm)**2, x_space))
        if norm > 0:
            psi_qm /= norm
        
        # Current width
        rho_qm_temp = np.abs(psi_qm)**2
        rho_qm_temp /= np.trapz(rho_qm_temp, x_space)
        mean_x = np.trapz(x_space * rho_qm_temp, x_space)
        sigma_x = np.sqrt(np.trapz((x_space - mean_x)**2 * rho_qm_temp, x_space))

        diff_sigma = abs(sigma_x - sigma_target)
        
        if diff_sigma < best_error:
            best_error = diff_sigma
            best_step = steps
            best_psi = psi_qm
        else:    
            print(f"\nSCHRÖDINGER COMPARISON\n")
            print(f"Convergence in {best_step} QM steps")
            print(f"Temporal ratio: τ_hydro/τ_QM = {CFG.N_steps/best_step:.2f}")
            break
        
        steps += 1
    
    rho_qm = np.abs(best_psi)**2
    rho_qm /= np.trapz(rho_qm, x_space)
 
    return rho_qm

# ===============================
# QUANTUM ANALYSIS
# ===============================

def analyze_results(data, theory):
    """
    Performs quantitative diagnostics on the simulation results.

    The analysis includes:
    • Correlation with Born-rule densities
    • L¹ distance errors
    • Comparison with fermionic and bosonic predictions
    • Pair correlation function g(r)
    • Emergent exclusion diagnostics

    No assumption of quantum statistics is made in the dynamics.
    """
    x_space = data['x_space']
    
    # ========================================
    # 1. PARTICLE 1
    # ========================================
    # vs |ψ1|²
    corr_p1_born = np.corrcoef(data['rho_p1_obs'], data['born_p1'])[0, 1]
    error_L1_p1_born = 0.5 * np.trapz(np.abs(data['rho_p1_obs'] - data['born_p1']), x_space)
    
    # vs Theoretical fermions
    corr_p1_fermion = np.corrcoef(data['rho_p1_obs'], theory['rho1_fermion'])[0, 1]
    error_L1_p1_fermion = 0.5 * np.trapz(np.abs(data['rho_p1_obs'] - theory['rho1_fermion']), x_space)
    
    # vs Theoretical bosons
    corr_p1_boson = np.corrcoef(data['rho_p1_obs'], theory['rho1_boson'])[0, 1]
    error_L1_p1_boson = 0.5 * np.trapz(np.abs(data['rho_p1_obs'] - theory['rho1_boson']), x_space)
    
    # ========================================
    # 2. PARTICLE 2
    # ========================================
    corr_p2_born = np.corrcoef(data['rho_p2_obs'], data['born_p2'])[0, 1]
    error_L1_p2_born = 0.5 * np.trapz(np.abs(data['rho_p2_obs'] - data['born_p2']), x_space)
    
    corr_p2_fermion = np.corrcoef(data['rho_p2_obs'], theory['rho2_fermion'])[0, 1]
    error_L1_p2_fermion = 0.5 * np.trapz(np.abs(data['rho_p2_obs'] - theory['rho2_fermion']), x_space)
    
    corr_p2_boson = np.corrcoef(data['rho_p2_obs'], theory['rho2_boson'])[0, 1]
    error_L1_p2_boson = 0.5 * np.trapz(np.abs(data['rho_p2_obs'] - theory['rho2_boson']), x_space)
    
    # ========================================
    # 3. TOTAL DENSITY 
    # ========================================
    corr_total_born = np.corrcoef(data['rho_total_obs'], data['born_sum'])[0, 1]
    error_L1_total_born = 0.5 * np.trapz(np.abs(data['rho_total_obs'] - data['born_sum']), x_space)
    
    corr_total_fermion = np.corrcoef(data['rho_total_obs'], theory['rho_total_fermion'])[0, 1]
    error_L1_total_fermion = 0.5 * np.trapz(np.abs(data['rho_total_obs'] - theory['rho_total_fermion']), x_space)
    
    corr_total_boson = np.corrcoef(data['rho_total_obs'], theory['rho_total_boson'])[0, 1]
    error_L1_total_boson = 0.5 * np.trapz(np.abs(data['rho_total_obs'] - theory['rho_total_boson']), x_space)
    
    # ========================================
    # 4. g(r) FONCTION
    # ========================================
    r_vals, g_r, hist_real_norm, hist_ghost_norm = compute_pair_correlation(
        data['hist_dist_real'], data['hist_dist_ghost'], data['r_centers']
    )
    g_0 = g_r[np.argmin(np.abs(r_vals - 1.0))]
    
    # Exclusion Factor
    idx_2 = np.where(r_vals < 5.0)[0]
    overlap_real = np.sum(data['hist_dist_real'][idx_2]) / np.sum(data['hist_dist_real'])
    overlap_ghost = np.sum(data['hist_dist_ghost'][idx_2]) / np.sum(data['hist_dist_ghost'])
    exclusion_factor = overlap_ghost / (overlap_real + 1e-9)
    
    # ========================================
    # 5. SCHRÖDINGER
    # ========================================
    # P1
    mean_x_p1 = np.trapz(x_space * data['born_p1'], x_space)
    sigma_x_p1 = np.sqrt(np.trapz((x_space - mean_x_p1)**2 * data['born_p1'], x_space))
    rho_qm_p1 = compare_schrodinger(x_space, sigma_x_p1)
    
    # P2
    mean_x_p2 = np.trapz(x_space * data['born_p2'], x_space)
    sigma_x_p2 = np.sqrt(np.trapz((x_space - mean_x_p2)**2 * data['born_p2'], x_space))
    rho_qm_p2 = compare_schrodinger(x_space, sigma_x_p2)
    
    # Total
    rho_qm_total = rho_qm_p1 + rho_qm_p2

    # Symmetry check
    print(f"\n5. WIDTHS :")
    print(f"   σ_P1    : {sigma_x_p1:.3f}")
    print(f"   σ_P2    : {sigma_x_p2:.3f}")
    # print(f"   σ_total : {sigma_x_total:.3f}")
    print(f"   Asymmetry : {abs(sigma_x_p1 - sigma_x_p2)/sigma_x_p1 * 100:.1f}%")

    # ========================================
    # DISPLAY
    # ========================================
    print("\n" + "="*70)
    print("QUANTITATIVE RESULTS")
    print("="*70)
    
    print("\n1. PARTICLE 1:")
    print(f"   vs |ψ₁|² observed: corr={corr_p1_born:.4f}, L¹={error_L1_p1_born:.5f}")
    print(f"   vs Theoretical fermion: corr={corr_p1_fermion:.4f}, L¹={error_L1_p1_fermion:.5f}")
    print(f"   vs Theoretical boson: corr={corr_p1_boson:.4f}, L¹={error_L1_p1_boson:.5f}")
    
    print("\n2. PARTICULE 2:")
    print(f"   vs |ψ₂|² observed: corr={corr_p2_born:.4f}, L¹={error_L1_p2_born:.5f}")
    print(f"   vs Theoretical fermion: corr={corr_p2_fermion:.4f}, L¹={error_L1_p2_fermion:.5f}")
    print(f"   vs Theoretical boson: corr={corr_p2_boson:.4f}, L¹={error_L1_p2_boson:.5f}")
    
    print("\n3. TOTAL DENSITY:")
    print(f"   vs |ψ₁+ψ₂|² observed: corr={corr_total_born:.4f}, L¹={error_L1_total_born:.5f}")
    print(f"   vs Theoretical fermion: corr={corr_total_fermion:.4f}, L¹={error_L1_total_fermion:.5f}")
    print(f"   vs Theoretical boson  : corr={corr_total_boson:.4f}, L¹={error_L1_total_boson:.5f}")
    
    print("\n4. PAIR CORRELATIONS:")
    print(f"   g(r≈1): {g_0:.3f}")
    print(f"   Exclusion Factor: {exclusion_factor:.2f}x")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC")
    print("="*70)
    
    # Born Convergence
    born_ok = (corr_p1_born > 0.95 and corr_p2_born > 0.95 and 
               error_L1_p1_born < 0.05 and error_L1_p2_born < 0.05)
    
    if born_ok and g_0 < 0.5 and exclusion_factor > 2.0:
        print("🎉 FERMIONIC BEHAVIOR CONFIRMED")
        print(f"   ✓ Born Convergence : ρ ≈ |ψ|² (corr>{0.95:.2f})")
        print(f"   ✓ Exclusion factor (real vs ghost) significantly enhanced")
        print(f"   ✓ Fermi Hole: g(1)={g_0:.3f} < 0.5")
    elif born_ok:
        print("✓ BORN CONVERGENCE BORN CONFIRMED")
        print("⚠️  Partial fermionic signature")
    else:
        print("❌ Insufficient fermionic signature")

    return {
        'corr_p1_born': corr_p1_born,
        'error_L1_p1_born': error_L1_p1_born,
        'corr_p1_fermion': corr_p1_fermion,
        'error_L1_p1_fermion': error_L1_p1_fermion,
        'corr_p1_boson': corr_p1_boson,
        'error_L1_p1_boson': error_L1_p1_boson,        
        'corr_p2_born': corr_p2_born,
        'error_L1_p2_born': error_L1_p2_born,
        'corr_p2_fermion': corr_p2_fermion,
        'error_L1_p2_fermion': error_L1_p2_fermion,   
        'corr_p2_boson': corr_p2_boson,
        'error_L1_p2_boson': error_L1_p2_boson,                
        'corr_total_born': corr_total_born,
        'error_L1_total_born': error_L1_total_born,
        'corr_total_fermion': corr_total_fermion,
        'error_L1_total_fermion': error_L1_total_fermion,
        'corr_total_boson': corr_total_boson,
        'error_L1_total_boson': error_L1_total_boson,        
        'g_r': (r_vals, g_r),
        'g_0': g_0,
        'exclusion_factor': exclusion_factor,
        'hist_real': hist_real_norm,
        'hist_ghost': hist_ghost_norm,
        'rho_qm_p1': rho_qm_p1,      
        'rho_qm_p2': rho_qm_p2,      
        'rho_qm_total': rho_qm_total        
    }
    
# ===============================
# VISUALIZATION (PAGE 1)
# ===============================

def plot_results_page1(data, theory, metrics):
    """
    Page 1:
    • One-particle densities
    • Comparison with Born rule
    • Fermionic and bosonic theoretical references
    • Residuals highlighting deviations
    """
    x = data['x_space']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ========================================
    # 1. PARTICLE 1 - Density
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    mean_pos_p1 = np.average(x, weights=data['rho_p1_obs'])
    ax1.plot(x, data['rho_p1_obs'], 'b-', lw=2.5, label='ρ₁ observed')
    ax1.plot(x, data['born_p1'], 'k--', lw=2, alpha=0.8, label='|ψ₁|² (Born)')
    ax1.plot(x, theory['rho1_fermion'], 'r:', lw=2, label='Theoretical fermion')
    ax1.plot(x+mean_pos_p1, theory['rho1_boson'], 'g:', lw=1.5, alpha=0.6, label='Theoretical boson')
    ax1.plot(x+mean_pos_p1, metrics['rho_qm_p1'], 'm:', lw=1.5, alpha=0.6, label='Schrödinger')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel('x Position')
    ax1.set_ylabel('Densitu')
    ax1.set_title(f'Particle 1 (corr_Born={metrics["corr_p1_born"]:.4f}, L¹={metrics["error_L1_p1_born"]:.5f})', 
                 fontweight='bold', fontsize=11)
    
    # ========================================
    # 2. PARTICLE 2 - Density
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    mean_pos_p2 = np.average(x, weights=data['rho_p2_obs'])
    ax2.plot(x, data['rho_p2_obs'], 'b-', lw=2.5, label='ρ₂ observed')
    ax2.plot(x, data['born_p2'], 'k--', lw=2, alpha=0.8, label='|ψ₂|² (Born)')
    ax2.plot(x, theory['rho2_fermion'], 'r:', lw=2, label='Theoretical fermion')
    ax2.plot(x+mean_pos_p2, theory['rho2_boson'], 'g:', lw=1.5, alpha=0.6, label='Theoretical boson')
    ax2.plot(x+mean_pos_p2, metrics['rho_qm_p2'], 'm:', lw=1.5, alpha=0.6, label='Schrödinger')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel('x Position')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Particle 2 (corr_Born={metrics["corr_p2_born"]:.4f}, L¹={metrics["error_L1_p2_born"]:.5f})', fontweight='bold', fontsize=11)
    
    # ========================================
    # 3. RESIDUALS P1
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    res_born = data['rho_p1_obs'] - data['born_p1']
    # res_fermion = data['rho_p1_obs'] - theory['rho1_fermion']
    ax3.plot(x, res_born, 'k-', lw=1.5, label='ρ - |ψ₁|²')
    # ax3.plot(x, res_fermion, 'r-', lw=1.5, alpha=0.7, label='ρ - Fermion')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between(x, 0, res_born, alpha=0.2, color='black')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlabel('x Position')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals P1', fontweight='bold', fontsize=11)
    
    # ========================================
    # 4. RESIDUALS P2
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    res_born_p2 = data['rho_p2_obs'] - data['born_p2']
    # res_fermion_p2 = data['rho_p2_obs'] - theory['rho2_fermion']
    ax4.plot(x, res_born_p2, 'k-', lw=1.5, label='ρ - |ψ₂|²')
    # ax4.plot(x, res_fermion_p2, 'r-', lw=1.5, alpha=0.7, label='ρ - Fermion')
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.fill_between(x, res_born_p2, alpha=0.2, color='black')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_xlabel('x Position')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals P2', fontweight='bold', fontsize=11)
    
    # ========================================
    # 5. TOTAL DENSITY 
    # ========================================
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(x, data['rho_total_obs'], 'b-', lw=2.5, label='ρ₁+ρ₂ observed')
    ax5.plot(x, data['born_sum'], 'k--', lw=2, alpha=0.8, label='|ψ₁+ψ₂|² (Born)')
    ax5.plot(x, theory['rho_total_fermion'], 'r:', lw=2, label='Theoretical fermion')
    ax5.plot(x, theory['rho_total_boson'], 'g:', lw=1.5, alpha=0.6, label='Theoretical boson')
    ax5.plot(x, metrics['rho_qm_total'], 'm:', lw=1.5, alpha=0.6, label='Schrödinger')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.set_xlabel('x Position', fontsize=11)
    ax5.set_ylabel('Total Density', fontsize=11)
    ax5.set_title(f'Total Density (corr_Born={metrics["corr_total_born"]:.4f}, L¹={metrics["error_L1_total_born"]:.5f})', 
                 fontweight='bold', fontsize=12)
    
    plt.suptitle('Page 1: Born Convergence and Comparison with Theories', 
                fontsize=14, fontweight='bold', y=0.995)
    
    base_name = f"Pauli_Exclusion_N{CFG.N_runs}_Page1"
    i = 1
    while os.path.exists(f"{base_name}_{i}.png"):
        i += 1
    filename = f"{base_name}_{i}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Page 1 saved: {filename}")

# ===============================
# VISUALIZATION (PAGE 2)
# ===============================

def plot_results_page2(data, theory, metrics):
    """
    Page 2:
    • Pair correlation function g(r)
    • Distance distributions (real vs ghost)
    • Joint position heatmap
    • Comparison with fermionic theoretical exclusion
    """
    x = data['x_space']
    r_vals, g_r = metrics['g_r']
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================
    # 1. g(r) FONCTION
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r_vals, g_r, 'r-', linewidth=2.5, label='g(r) mesured')
    ax1.axhline(1, color='gray', linestyle='--', linewidth=1.5, label='Independent particles')
    ax1.fill_between(r_vals, g_r, 1, where=(g_r < 1), 
                     facecolor='red', alpha=0.2, label='Fermi Hole')
    ax1.set_xlabel('Distance r', fontsize=11)
    ax1.set_ylabel('g(r)', fontsize=11)
    ax1.set_title(f'Pair correlation (g(1)={metrics["g_0"]:.3f})', 
                 fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 2)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # ========================================
    # 2. DISTANCES DISTRIBUTION
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(r_vals, metrics['hist_ghost'], 'gray', lw=2, alpha=0.6, label='Ghosts')
    ax2.fill_between(r_vals, metrics['hist_ghost'], alpha=0.3, color='gray')
    ax2.plot(r_vals, metrics['hist_real'], 'r-', lw=2.5, label='Real')
    ax2.fill_between(r_vals, metrics['hist_real'], alpha=0.2, color='red')
    ax2.set_xlabel('Distance |x₁ - x₂|', fontsize=11)
    ax2.set_ylabel('Probability density', fontsize=11)
    ax2.set_title(f'Distances distribution (Excl: {metrics["exclusion_factor"]:.2f}x)', 
                 fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 40)
    
    # ========================================
    # 3. HEATMAP 2D : Position(P1) vs Position(P2)
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    H = data['hist_2d']
    extent = [CFG.x_min, CFG.x_max, CFG.x_min, CFG.x_max]
    im = ax3.imshow(H.T, origin='lower', extent=extent, cmap='hot', aspect='auto', 
                    interpolation='bilinear')
    
    ax3.plot([CFG.x_min, CFG.x_max], [CFG.x_min, CFG.x_max], 'cyan', linewidth=2.5, 
             linestyle='--', label='x₁=x₂')
    
    ax3.set_xlabel('Position P1', fontsize=11)
    ax3.set_ylabel('Position P2', fontsize=11)
    ax3.set_title('Joined positions Heatmap', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=10, loc='upper left')
    plt.colorbar(im, ax=ax3, label='Density')
    
    # ========================================
    # 4. THEORETICAL FERMIONS HEATMAP
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    rho_2d = theory['rho_2d_fermion']
    extent_theory = [CFG.x_min, CFG.x_max, CFG.x_min, CFG.x_max]
    im2 = ax4.imshow(rho_2d.T, origin='lower', extent=extent_theory, cmap='hot', 
                     aspect='auto', interpolation='bilinear')
    
    # Theoretical diagonal
    ax4.plot([CFG.x_min, CFG.x_max], [CFG.x_min, CFG.x_max], 'cyan', linewidth=2.5, 
             linestyle='--', label='x₁=x₂')
    
    ax4.set_xlabel('Position P1', fontsize=11)
    ax4.set_ylabel('Position P2', fontsize=11)
    ax4.set_title('Theoretical fermion |ψ₋(x₁,x₂)|²', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=10, loc='upper left')
    plt.colorbar(im2, ax=ax4, label='Theoretical density')
    
    plt.suptitle('Page 2: Spatial Correlations and Pauli Exclusion', 
                fontsize=14, fontweight='bold', y=0.995)
    
    base_name = f"Pauli_Exclusion_N{CFG.N_runs}_Page2"
    i = 1
    while os.path.exists(f"{base_name}_{i}.png"):
        i += 1
    filename = f"{base_name}_{i}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Page 2 saved: {filename}")

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    # 1. Simulation
    data = run_pauli_simulation()
    
    # 2. Calculation of theoretical densities
    print("\n🔬 Calculation of theoretical densities...")
    theory = compute_theoretical_densities(
        data['x_space'], data['phi1_mean'], data['phi2_mean']
    )
    
    # 3. Quantitative analysis
    metrics = analyze_results(data, theory)
    
    # 4. Visualization (2 pages)
    plot_results_page1(data, theory, metrics)
    plot_results_page2(data, theory, metrics)
    
    print("\n" + "="*70)
    print("✓ SIMULATION COMPLETED")
    print("="*70)
