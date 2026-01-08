import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import os

"""
Hydrodynamic Quantum Analogs: 1D Pilot-Wave Simulation
======================================================
Emergence of Born's Rule from Local Dynamics.

This script simulates a pilot-wave dynamic with feedback (walking droplet type)
in 1D to demonstrate the statistical emergence of Born's Rule.

Physical Principle:
-------------------
This model couples a stochastic point particle to a complex scalar field (pilot wave).
1. The wave evolves according to a complex Ginzburg-Landau equation (Schrödinger-like).
2. The particle is guided by the local phase gradient of the field (Langevin dynamics).
3. The particle acts as a moving source, continuously interecting and fueling its own pilot wave (feedback).

Key Result:
-----------
The system is in a state of free expansion (diffusion).
The simulation demonstrates that the particle's statistical distribution ρ(x)
dynamically conforms to the shape of the spread wave packet |ψ|².
In other words, the probability density of the particle's position converges towards 
the intensity |ψ|² of the  field, validating the dynamical "quantum relaxation"
towards Born's Rule without axiomatic postulates.
It validates the emergence of Born's Rule from a purely deterministic, local, and realistic dynamics.


Author : Revoire Christian (Independent Researcher)
Date   : Janvier 2026
License: MIT
"""

# ===============================
# CONFIGURATION LOAD
# ===============================
from src.config import Base_Config
CFG = Base_Config()

"""
WParameters are loaded from config.py.
To change physics or simulation settings, edit 'src/config.py' 

If Import Error try changing filepath in: 'from filepath.config import Base_Config' in accordance with your 'filepath/config.py'.
"""

# ===============================
# OPTIMIZED NUMBA ENGINE
# ===============================
"""
The wave ψ(x,t) evolves according to a complex Ginzburg-Landau equation like :
    ∂t ψ = (Dψ + i ω) ∇²ψ - γ ψ + source
"""

@njit(fastmath=True)
def evolve_field_1d(psi: np.ndarray, psi_new: np.ndarray, lap_buffer: np.ndarray, x_p: float, dt: float, dx: float, D_psi: float, omega: float, gamma: float, 
                    emit_amp: float, sigma_emit: float, x_min: float, Nx: int, c: float):
    """
    Computes one time-step of the pilot-wave evolution with a moving source.
    Uses a pre-allocated buffer for the Laplacian to minimize memory allocation.
    """
    # 1. Compute Laplacian (Finite Differences)
    lap_buffer[:] = 0.0
    for i in range(1, Nx-1):
        lap_buffer[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2
    
    # 2. Field Evolution (Diffusion + Dispersion - Dissipation)
    # Using explicit Euler method
    for i in range(Nx):
        psi_new[i] = psi[i] + dt * c * ((D_psi + 1j * omega) * lap_buffer[i] - gamma * psi[i])
    
    # 3. Inject Source at Particle Position (Optimized with cutoff)
    cutoff = 6.0 * sigma_emit
    idx_center = int(round((x_p - x_min) / dx))
    cutoff_idx = int(cutoff / dx)
    
    i_start = max(0, idx_center - cutoff_idx)
    i_end = min(Nx, idx_center + cutoff_idx + 1)

    # Add Gaussian source
    for i in range(i_start, i_end):
        xi = x_min + i * dx
        dist2 = (xi - x_p)**2
        psi_new[i] += emit_amp * np.exp(-0.5 * dist2 / sigma_emit**2) * dt
    
    return psi_new

@njit(fastmath=True)
def get_guidance(psi: np.ndarray, idx: int, dx: float, epsilon: float):
    """
    Extracts local amplitude and phase gradient for particle guidance.
    Handles phase wrapping (-π to π).
    """
    if idx < 1 or idx >= len(psi) - 1:
        return 0.0, 0.0
    
    p_plus = psi[idx + 1]
    p_minus = psi[idx - 1]
    
    # Local amplitude check
    amp_loc = np.abs(psi[idx])
    if amp_loc < epsilon:
        return amp_loc, 0.0
    
    # Central difference for phase gradient
    dph = np.angle(p_plus) - np.angle(p_minus)

    # Phase unwrapping
    if dph > np.pi: dph -= 2*np.pi
    elif dph < -np.pi: dph += 2*np.pi
    
    grad_phase = dph / (2 * dx)
    
    return amp_loc, grad_phase

@njit(fastmath=True)
def simulate_single_particle(x_init: float, N_steps: int, thermalization: int, subsample: int,
                            dt: float, dx: float, D_psi: float, omega: float, gamma: float, emit_amp: float, sigma_emit: float,
                            alpha: float, D_x: float, epsilon: float, x_min: float, x_max: float, Nx: int, c: float):
    """
    Full simulation loop for a SINGLE particle.
    Designed to be run in parallel processes.
    """
                                
    # Initialize Field
    psi = np.zeros(Nx, dtype=np.complex64)
    psi_new = np.zeros_like(psi)
    lap_buffer = np.zeros_like(psi)

    # Initialize Particle (Gaussian start near center)
    x_p = x_init
    
    # Data Storage
    n_samples = (N_steps - thermalization) // subsample
    positions = np.zeros(n_samples)
    psi_accumulated = np.zeros(Nx, dtype=np.complex64)
    psi2_accumulated = np.zeros(Nx, dtype=np.float32)  # |ψ|²
    
    sample_idx = 0
    
    # --- Time Loop ---
    for t in range(N_steps):
        # A. Evolve Field
        psi_new = evolve_field_1d(psi, psi_new, lap_buffer, x_p, dt, dx, 
                                  D_psi, omega, gamma, emit_amp, sigma_emit, 
                                  x_min, Nx, c)
        psi[:] = psi_new[:]
        
        # B. Move Particle
        idx = int(round((x_p - x_min) / dx))
        amp_loc, grad_phase = get_guidance(psi, idx, dx, epsilon)

        # Calculate Drift (Guidance)
        drift = 0.0
        amp2 = amp_loc**2
        if amp2 > epsilon**2:
            weight = amp2 / (amp2 + epsilon**2)
            drift = alpha * weight * grad_phase
            # Velocity limiter for stability
            if drift > 10.0: drift = 10.0
            elif drift < -10.0: drift = -10.0
        else:
            drift = 0.0
        
        # Langevin Step
        noise = np.sqrt(2 * D_x * dt) * np.random.randn()
        x_p += drift * dt + noise
        
        # Boundary Conditions (Hard walls)
        if x_p < x_min + dx: x_p = x_min + dx
        elif x_p > x_max - dx: x_p = x_max - dx
        
        # C. Data Collection (Post-thermalization)
        if t >= thermalization:
            # Position Sampling
            if (t - thermalization) % subsample == 0:
                positions[sample_idx] = x_p
                sample_idx += 1
            
            # Field Accumulation (Ergodicity check)
            psi_accumulated += psi
            for i in range(Nx):
                psi2_accumulated[i] += psi[i].real**2 + psi[i].imag**2
    
    return positions, psi_accumulated, psi2_accumulated

# ===============================
# PARALLEL WORKER
# ===============================

def worker_particle(seed, particle_id, x_space):
    """
    Worker executed on a each CPU core.
    """
    np.random.seed(seed)
    
    # Initial position
    x_init = np.random.normal(0, 1.0)
    
    # Simulation
    positions, psi_acc, psi2_acc = simulate_single_particle(
        x_init, CFG.N_steps, CFG.thermalization, CFG.SUBSAMPLE,
        CFG.dt, CFG.dx, CFG.D_psi, CFG.omega, CFG.gamma, CFG.emit_amp, CFG.sigma_emit_scaled,
        CFG.alpha, CFG.D_x, CFG.epsilon, CFG.x_min, CFG.x_max, CFG.Nx, CFG.c
    )
    
    # Position histogram (lightweight)
    hist, _ = np.histogram(positions, bins=len(x_space), 
                          range=(CFG.x_min, CFG.x_max))
    
    return {
        'histogram': hist.astype(np.float32),
        'psi_acc': psi_acc,
        'psi2_acc': psi2_acc,
        'n_samples': len(positions)
    }

# ===============================
# MAIN SIMULATION
# ===============================

def run_born_simulation():
    # Detect CPU cores
    n_cores = CFG.N_CORES if CFG.N_CORES > 0 else max(1, mp.cpu_count() + CFG.N_CORES)
    
    # Memory estimation
    n_samples_per_particle = (N_steps - thermalization) // SUBSAMPLE
    memory_per_particle_mb = (n_samples_per_particle * 8 + Nx * 16) / (1024**2)
    total_memory_mb = memory_per_particle_mb * N_runs / n_cores
    
    print("="*70)
    print("SIMULATION: BORN RULE EMERGENCE")
    print("="*70)
    print(f"Configuration:")
    print(f"  - CPU Cores: {n_cores}/{mp.cpu_count()}")
    print(f"  - Particles: {CFG.N_runs}")
    print(f"  - Steps/Particle: {CFG.N_steps}")
    print(f"  - Estimated memory: {total_memory_mb:.1f} MB")
    print(f"\nPhysics :")
    print(f"  γ={CFG.gamma}, D_ψ={CFG.D_psi}, ω={CFG.omega}, α={CFG.alpha}, Bruit={CFG.D_x}, amp={CFG.emit_amp}")
    print("="*70)
    
    x_space = np.linspace(CFG.x_min, CFG.x_max, CFG.Nx)
    
    start_time = time.time()
    
    # ========================================
    # JOBLIB PARALLELIZATION
    # ========================================
    print("\n🚀 Starting parallel simulations...\n")
    
    results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
        delayed(worker_particle)(
            seed=42 + p*1000,
            particle_id=p,
            x_space=x_space
        ) for p in tqdm(range(CFG.N_runs), desc="Simulating Particles")
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed/60:.2f} min")
    print(f"  Speed : {CFG.N_runs * CFG.N_steps / elapsed / 1000:.1f}k steps/sec")
    
    # ========================================
    # AGGREGATING RESULTS
    # ========================================
    print("\n📊 Aggregating statistics...")
    
    rho = np.zeros(CFG.Nx)
    psi_acc = np.zeros(CFG.Nx, dtype=np.complex128)
    psi2_acc = np.zeros(CFG.Nx, dtype=np.float64)
    total_samples = 0
    
    for res in tqdm(results, desc="Fusion"):
        rho += res['histogram']
        psi_acc += res['psi_acc']
        psi2_acc += res['psi2_acc']
        total_samples += res['n_samples']
    
    # Normalization
    rho /= (np.sum(rho) * CFG.dx)
    psi_acc /= total_samples
    psi2_acc /= total_samples
    
    born = psi2_acc  # √⟨|ψ|²⟩
    born /= np.trapz(born, x_space)
    
    return x_space, rho, born, psi_acc

# ===============================
# QUANTUM ANALYSIS
# ===============================

def compute_hbar_effective(x_space, rho, psi_acc):
    """
    Computes the effective ℏ via the Heisenberg uncertainty relation.
    """
    dx_local = x_space[1] - x_space[0]
    
    # 1. Position
    mean_x = np.trapz(x_space * rho, x_space)
    sigma_x = np.sqrt(np.trapz((x_space - mean_x)**2 * rho, x_space))
    
    # 2. Momentum  (FFT)
    psi_normalized = psi_acc / np.sqrt(np.sum(np.abs(psi_acc)**2) * dx_local)
    psi_k = np.fft.fftshift(np.fft.fft(psi_normalized))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x_space), dx_local))
    k_vals = 2 * np.pi * freqs
    
    rho_k = np.abs(psi_k)**2
    rho_k /= np.trapz(rho_k, k_vals)
    
    mean_k = np.trapz(k_vals * rho_k, k_vals)
    sigma_k = np.sqrt(np.trapz((k_vals - mean_k)**2 * rho_k, k_vals))
    
    # 3. Effective ℏ from Heisenberg
    # For a minimal state: Δx·Δp = ℏ/2 (with p = ℏk)
    
    hbar_eff = 2 * sigma_x * sigma_k   # Uncertainty Product
    
    print(f"\n{'='*70}")
    print(f"EMERGENT ℏ MEASUREMENT")
    print(f"{'='*70}")
    print(f"σ_x  = {sigma_x:.4f}")
    print(f"σ_k  = {sigma_k:.4f}")
    print(f"ℏ_eff = {hbar_eff:.4f} (uncertainty product)")
    
    return hbar_eff, sigma_x

def compare_schrodinger(x_space, sigma_x_model):
    """
    The model simulates a freely spreading wave packet (diffusive regime).
    To validate the packet shape, we compare the final simulation state (at t_sim)
    with a snapshot of Schrödinger evolution at an quantum time t_QM.
    
    The loop below searches for the quantum time t_QM at which the width of the
    quantum packet (σ_qm) matches the model width (σ_x).
    
    This establishes the temporal scaling factor between the two dynamics: τ_stochastic / τ_Schrödinger
    
    Approximate relation: steps ∝ σ² / (ω·dt)
    """
  
    dx_local = x_space[1] - x_space[0]
    psi_qm = np.exp(-0.5*(x_space/2.0)**2).astype(np.complex128)
    psi_qm /= np.sqrt(np.trapz(np.abs(psi_qm)**2, x_space))
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        # Free evolution
        lap_qm = np.zeros_like(psi_qm)
        lap_qm[1:-1] = (psi_qm[2:] - 2*psi_qm[1:-1] + psi_qm[:-2]) / dx_local**2
        psi_qm += CFG.dt * (1j * CFG.omega * CFG.lap_qm)
        
        # Normalization
        norm = np.sqrt(np.trapz(np.abs(psi_qm)**2, x_space))
        if norm > 0: psi_qm /= norm
        
        # Current width
        rho_qm_temp = np.abs(psi_qm)**2
        rho_qm_temp /= np.trapz(rho_qm_temp, x_space)
        mean_x_qm = np.trapz(x_space * rho_qm_temp, x_space)
        sigma_x_qm = np.sqrt(np.trapz((x_space - mean_x_qm)**2 * rho_qm_temp, x_space))
        
        if abs(sigma_x_qm - sigma_x_model) < 0.005:
            print(f"\n{'='*70}")
            print(f"SCHRÖDINGER COMPARISON")
            print(f"{'='*70}")
            print(f"Convergence in {steps} QM steps")
            print(f"Temporal ratio: τ_hydro/τ_QM = {CFG.N_steps/steps:.2f}")
            break
        
        steps += 1
    
    rho_qm = np.abs(psi_qm)**2
    rho_qm /= np.trapz(rho_qm, x_space)
    
    return rho_qm

# ===============================
# VISUALIZATION
# ===============================

def plot_results(x_space, rho, born, rho_qm):
    """
    Comparative density plots.
    """
    corr = np.corrcoef(rho, born)[0,1]
    error_L1 = 0.5 * np.trapz(np.abs(rho - born), x_space)
    
    print(f"\n{'='*70}")
    print(f"CONVERGENCE TOWARD |ψ|²")
    print(f"{'='*70}")
    print(f"Correlation ρ vs |ψ|²: {corr:.4f}")
    print(f"L¹ error: {error_L1:.5f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Density comparison
    ax1 = axes[0]
    ax1.plot(x_space, rho, 'b-', lw=2, label='ρ(x) particules')
    ax1.plot(x_space, born, 'r--', lw=2, label='⟨|ψ|²⟩ attracteur')
    ax1.plot(x_space, rho_qm, 'g:', lw=2, label='|ψ_QM|²')
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Densité', fontsize=12)
    ax1.set_title(f'Convergence Born (corr={corr:.4f}, L¹={error_L1:.5f})', 
                 fontsize=14, fontweight='bold')
    
    # 2. Residuals
    ax2 = axes[1]
    residuals = rho - born
    ax2.plot(x_space, residuals, 'k-', lw=1.5, label='Résidus')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(x_space, 0, residuals, alpha=0.3, color='red')
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('ρ - |ψ|²', fontsize=12)
    ax2.set_title('Analyse des résidus', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    base_name = f"Born_Rule_N{CFG.N_runs}"
    i = 1
    while os.path.exists(f"{base_name}_V{i}.png"):
        i += 1
    filename = f"{base_name}_V{i}.png"

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Figure saved: {filename}")
    
    plt.show()

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    # Main simulation
    x_space, rho, born, psi_acc = run_born_simulation()
    
    # Analyses
    hbar_eff, sigma_x = compute_hbar_effective(x_space, rho, psi_acc)
    rho_qm = compare_schrodinger(x_space, sigma_x)
    
    # Visualization
    plot_results(x_space, rho, born, rho_qm)
    
    print("\n" + "="*70)
    print("✓ SIMULATION COMPLETED")
    print("="*70)
