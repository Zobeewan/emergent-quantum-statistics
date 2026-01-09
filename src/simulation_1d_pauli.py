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
==========================================
Emergence of Pauli Exclusion Principle from Local Dynamics. 

This simulation investigates the emergence of fermionic behavior
(Pauli exclusion) from a stochastic field–particle dynamics model.

The model extends a previously validated framework showing
the emergence of the Born rule (ρ ≈ |ψ|²) for single particles,
and now focuses on two-particle correlations.

Physical Principle:
-------------------
• Each particle generates its own complex guiding field ψ(x,t).
• The particles follow a stochastic drift driven by the phase gradient.
  of a combined guiding field, the sum of the 2 fields generated.
• Symmetric or antisymmetric coupling is enforced dynamically.
• No explicit antisymmetrization of particle trajectories.


Key Result:
-----------
• Pair correlation functions g(r → 0) < 1
• The joined positions heatmap show a forbidden diagonal for x1 = x2
• The distinct prediction in the joined positions come from the fermion indistinguishability nature 
• Single particles statistical distribution ρ(x) still dynamically conforms to the shape of the spread wave packet |ψ|².
• Both particles statistical distribution ρ(x) dynamically conforms to the shape of a symmetric coupling of fields (boson's coupling).
• Pauli exclusion seem to emerges from local field dynamics acting as a repulsive force 


Author : Revoire Christian (Independent Researcher)
Date   : Janvier 2026
License: MIT
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

# --- Initial configuration of particle positions --- 
CFG.SIDE = "norm"      # "norm" = Particle 1 on the left, Particle 2 on the right
                       # "rand" = Random                   
                       # anything else starts inverted

# --- Coupling ---
CFG.coupling_type = "sum"    # "sum" →  |ψ|+|ψ| but fermionic behavior via effective repulsion
                             # "diff" → |ψ|-|ψ| but bosonic behavior via effective attraction

# ===============================
# NOYAU PHYSIQUE (Numba)
# ===============================

@njit(fastmath=True)
def get_drift(psi_val, psi_prev, psi_next, dx, epsilon, alpha):
    """
    Computes the local drift velocity from the phase gradient
    of the guiding field.

    This is the core guidance law of the model.
    """
  
    amp2 = np.abs(psi_val)**2
    if amp2 < epsilon**2:
        return 0.0
      
    # Prevents ill-defined phase when |ψ| ≈ 0
    if not np.isfinite(psi_prev) or not np.isfinite(psi_next) or not np.isfinite(psi_val):
        return 0.0

    amp_prev = np.abs(psi_prev)
    amp_next = np.abs(psi_next)

    if amp_prev < epsilon or amp_next < epsilon:
        return 0.0

    dph = np.angle(psi_next) - np.angle(psi_prev)

    # Phase unwrapping
    if dph > np.pi: dph -= 2*np.pi
    elif dph < -np.pi: dph += 2*np.pi
    
    if not np.isfinite(dph):
        return 0.0
    
    grad_phase = dph / (2 * dx)
    weight = amp2 / (amp2 + epsilon**2)
    drift = alpha * weight * grad_phase
    
    if not np.isfinite(drift):
        return 0.0

    if drift > 10.0: drift = 10.0
    elif drift < -10.0: drift = -10.0
    
    return drift

@njit(fastmath=True)
def evolve_field(psi, x_particle, dt, dx, D_psi, omega, gamma, 
                emit_amp, sigma_emit, x_min, Nx):
    lap = np.zeros_like(psi)
    for i in range(1, Nx-1):
        lap[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2
    
    psi_new = psi + dt * c * ((D_psi + 1j * omega) * lap - gamma * psi)
    
    cutoff = 6.0 * sigma_emit
    cutoff_idx = int(cutoff / dx)
    idx = int(round((x_particle - x_min) / dx))
    
    i_start = max(0, idx - cutoff_idx)
    i_end = min(Nx, idx + cutoff_idx + 1)
    
    for i in range(i_start, i_end):
        xi = x_min + i * dx
        dist2 = (xi - x_particle)**2
        psi_new[i] += emit_amp * np.exp(-0.5 * dist2 / sigma_emit**2) * dt
    
    return psi_new

@njit(fastmath=True)
def simulate_interaction(x1_init, x2_init, N_steps, dt, dx, D_psi, omega, gamma,
                        emit_amp, sigma_emit, alpha, D_x, epsilon, 
                        x_min, x_max, Nx, coupling, thermalization):
    psi1 = np.zeros(Nx, dtype=np.complex128)
    psi2 = np.zeros(Nx, dtype=np.complex128)
    
    x1, x2 = x1_init, x2_init
    traj1 = np.zeros(N_steps)
    traj2 = np.zeros(N_steps)
    
    # Accumulation des champs après thermalisation
    psi1_acc = np.zeros(Nx, dtype=np.complex128)
    psi2_acc = np.zeros(Nx, dtype=np.complex128)
    psi_sum_acc = np.zeros(Nx, dtype=np.complex128)

    born1_acc = np.zeros(Nx, dtype=np.float64)
    born2_acc = np.zeros(Nx, dtype=np.float64)
    born_sum_acc = np.zeros(Nx, dtype=np.float64)

    n_acc = 0
    
    for t in range(N_steps):
        psi1 = evolve_field(psi1, x1, dt, dx, D_psi, omega, gamma,
                           emit_amp, sigma_emit, x_min, Nx)
        psi2 = evolve_field(psi2, x2, dt, dx, D_psi, omega, gamma,
                           emit_amp, sigma_emit, x_min, Nx)
        
        if coupling == 1:
            psi_guide = psi1 + psi2
        else:
            psi_guide = psi1 - psi2
        
        idx1 = int(round((x1 - x_min) / dx))
        idx2 = int(round((x2 - x_min) / dx))
        
        d1, d2 = 0.0, 0.0
        if 1 < idx1 < Nx-2:
            d1 = get_drift(psi_guide[idx1], psi_guide[idx1-1], 
                          psi_guide[idx1+1], dx, epsilon, alpha)
        if 1 < idx2 < Nx-2:
            d2 = get_drift(psi_guide[idx2], psi_guide[idx2-1],
                          psi_guide[idx2+1], dx, epsilon, alpha)
        
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
        
        # Accumulation après thermalisation
        if t >= thermalization:
            psi1_acc += psi1
            psi2_acc += psi2
            psi_sum_acc += psi_guide
            born1_acc += np.abs(psi1)**2
            born2_acc += np.abs(psi2)**2
            born_sum_acc += np.abs(psi_guide)**2
            n_acc += 1
    
    # Normalisation
    if n_acc > 0:
        psi1_acc /= n_acc
        psi2_acc /= n_acc
        psi_sum_acc /= n_acc
        born1_acc /= n_acc
        born2_acc /= n_acc
        born_sum_acc /= n_acc
    
    return traj1, traj2, psi1_acc, psi2_acc, psi_sum_acc, born1_acc, born2_acc, born_sum_acc

@njit(fastmath=True)
def simulate_solo(x_init, N_steps, dt, dx, D_psi, omega, gamma,
                 emit_amp, sigma_emit, alpha, D_x, epsilon,
                 x_min, x_max, Nx, thermalization):
    psi = np.zeros(Nx, dtype=np.complex128)
    x = x_init
    traj = np.zeros(N_steps)
    
    # Accumulation du champ
    psi_accumulated = np.zeros(Nx, dtype=np.complex128)
    born_accumulated = np.zeros(Nx, dtype=np.float64)
    n_acc = 0
    
    for t in range(N_steps):
        psi = evolve_field(psi, x, dt, dx, D_psi, omega, gamma,
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
        
        # Accumulation après thermalisation
        if t >= thermalization:
            psi_accumulated += psi
            born_accumulated += np.abs(psi)**2
            n_acc += 1
    
    # Normalisation
    if n_acc > 0:
        psi_accumulated /= n_acc
        born_accumulated /= n_acc
    
    return traj, psi_accumulated, born_accumulated

# ===============================
# FONCTION g(r)
# ===============================

def compute_pair_correlation(distances_real, distances_ghost, x_min, x_max, bins=100):
    L = x_max - x_min
    bin_edges = np.linspace(0, L, bins+1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    hist_real, _ = np.histogram(distances_real, bins=bin_edges, density=True)
    hist_ghost, _ = np.histogram(distances_ghost, bins=bin_edges, density=True)
    
    g_r = np.divide(hist_real, hist_ghost, 
                    out=np.ones_like(hist_real), 
                    where=(hist_ghost > 1e-10))
    
    return r_centers, g_r, hist_real, hist_ghost

# ===============================
# WORKER PARALLÈLE
# ===============================

def worker_particle(seed, particle_id, x_space, coupling_code, side):
    np.random.seed(seed)
    
    if side == "rand":
        # Tirage aléatoire : 50% normal, 50% inversé
        if np.random.rand() < 0.5:
            area_p1 = (-15.0, -5.0)  # P1 gauche
            area_p2 = (5.0, 15.0)    # P2 droite
        else:
            area_p1 = (5.0, 15.0)    # P1 droite
            area_p2 = (-15.0, -5.0)  # P2 gauche
    else:
        # Utilise les zones globales
        area_p1 = start_area_p1
        area_p2 = start_area_p2
    
    x1_init = np.random.uniform(area_p1[0], area_p1[1])
    x2_init = np.random.uniform(area_p2[0], area_p2[1])
    
    # 1. INTERACTION (avec accumulation des champs)
    t1, t2, psi1_acc, psi2_acc, psi_sum_acc, born1_acc, born2_acc, born_sum_acc = simulate_interaction(
        x1_init, x2_init, N_steps, dt, dx, D_psi, omega, gamma,
        emit_amp, sigma_emit, alpha, D_x, epsilon,
        x_min, x_max, Nx, coupling_code, thermalization
    )
    
    # 2. SOLO P1 (avec accumulation)
    ts1, phi1_acc, born1_accumulated = simulate_solo(
        x1_init, N_steps, dt, dx, D_psi, omega, gamma,
        emit_amp, sigma_emit, alpha, D_x, epsilon,
        x_min, x_max, Nx, thermalization
    )
    
    # 3. SOLO P2 (avec accumulation)
    ts2, phi2_acc, born2_accumulated = simulate_solo(
        x2_init, N_steps, dt, dx, D_psi, omega, gamma,
        emit_amp, sigma_emit, alpha, D_x, epsilon,
        x_min, x_max, Nx, thermalization
    )
    
    # Analyse post-thermalisation
    t1_post = t1[thermalization:]
    t2_post = t2[thermalization:]
    ts1_post = ts1[thermalization:]
    ts2_post = ts2[thermalization:]
    
    # Histogrammes des positions
    hist_p1_real, _ = np.histogram(t1_post, bins=len(x_space), range=(x_min, x_max))
    hist_p2_real, _ = np.histogram(t2_post, bins=len(x_space), range=(x_min, x_max))
    
    # Distances
    d_real = np.abs(t1_post - t2_post)
    d_ghost = np.abs(ts1_post - ts2_post)
    
    # Pour heatmap
    positions_p1 = t1_post[::10]  # Sous-échantillonnage pour heatmap
    positions_p2 = t2_post[::10]
    
    return {
        'hist_p1': hist_p1_real.astype(np.float32),
        'hist_p2': hist_p2_real.astype(np.float32),
        'psi1_acc': psi1_acc,
        'psi2_acc': psi2_acc,
        'psi_sum_acc': psi_sum_acc,
        'phi1_acc': phi1_acc,
        'phi2_acc': phi2_acc,
        'born1_acc': born1_acc, 
        'born2_acc': born2_acc,
        'born_sum_acc': born_sum_acc,
        'born1_accumulated': born1_accumulated, 
        'born2_accumulated': born2_accumulated,        
        'dist_real': d_real,
        'dist_ghost': d_ghost,
        'positions_p1': positions_p1,
        'positions_p2': positions_p2
    }

# ===============================
# SIMULATION PRINCIPALE
# ===============================

def run_pauli_simulation():
    n_cores = N_CORES if N_CORES > 0 else max(1, mp.cpu_count() + N_CORES)
    
    print("="*70)
    print("PAULI EXCLUSION - VERSION QUANTITATIVE v4")
    print("="*70)
    print(f"Configuration :")
    print(f"  - Cœurs CPU : {n_cores}/{mp.cpu_count()}")
    print(f"  - Particules : {N_runs} paires")
    print(f"  - Steps/paire : {N_steps}")
    print(f"\nParamètres physiques :")
    print(f"  γ={gamma}, D_ψ={D_psi}, ω={omega}, α={alpha}, D_x={D_x}")
    print(f"  Couplage : {coupling_type}")
    print("="*70)
    
    x_space = np.linspace(x_min, x_max, Nx)
    coupling_code = 1 if coupling_type == "sum" else 0
    
    start_time = time.time()
    
    # ========================================
    # PARALLÉLISATION
    # ========================================
    print("\n🚀 Lancement des simulations parallèles...\n")
    
    results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
        delayed(worker_particle)(
            seed = 42 + p*1000,
            particle_id = p,
            x_space = x_space,
            coupling_code = coupling_code,
            side = SIDE
        ) for p in tqdm(range(N_runs), desc="Paires")
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Simulation terminée en {elapsed/60:.2f} min")
    
    # ========================================
    # AGRÉGATION
    # ========================================
    print("\n📊 Agrégation des statistiques...")
    
    hist_p1_total = np.zeros(Nx)
    hist_p2_total = np.zeros(Nx)
    psi1_mean = np.zeros(Nx, dtype=np.complex128)
    psi2_mean = np.zeros(Nx, dtype=np.complex128)
    psi_sum_mean = np.zeros(Nx, dtype=np.complex128)
    phi1_mean = np.zeros(Nx, dtype=np.complex128)
    phi2_mean = np.zeros(Nx, dtype=np.complex128)
    born1_mean = np.zeros(Nx, dtype=np.float64)
    born2_mean = np.zeros(Nx, dtype=np.float64)
    born_sum_mean = np.zeros(Nx, dtype=np.float64)
    integral_born = np.zeros(Nx, dtype=np.float64)
    born1_accumulated_mean = np.zeros(Nx, dtype=np.float64)
    born2_accumulated_mean = np.zeros(Nx, dtype=np.float64)
    dist_real_all = []
    dist_ghost_all = []
    positions_p1_all = []
    positions_p2_all = []
    
    for res in tqdm(results, desc="Fusion"):
        hist_p1_total += res['hist_p1']
        hist_p2_total += res['hist_p2']
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
        dist_real_all.extend(res['dist_real'])
        dist_ghost_all.extend(res['dist_ghost'])
        positions_p1_all.extend(res['positions_p1'])
        positions_p2_all.extend(res['positions_p2'])
    
    # Normalisation
    psi1_mean /= N_runs
    psi2_mean /= N_runs
    psi_sum_mean /= N_runs
    phi1_mean /= N_runs
    phi2_mean /= N_runs
    born1_mean /= N_runs
    born2_mean /= N_runs
    born_sum_mean /= N_runs
    born1_accumulated_mean /= N_runs
    born2_accumulated_mean /= N_runs
    
    # Densités observées
    rho_p1_obs = hist_p1_total / (np.sum(hist_p1_total) * dx)
    rho_p2_obs = hist_p2_total / (np.sum(hist_p2_total) * dx)
    rho_total_obs = rho_p1_obs + rho_p2_obs
    
    # |ψ|² observés
    born_p1 = born1_mean                    
    born_p1 /= np.trapz(born_p1, x_space)
    
    born_p2 = born2_mean                    
    born_p2 /= np.trapz(born_p2, x_space)
        
    integral_born = np.trapz(born_sum_mean, x_space)        
    born_sum = 2 * (born_sum_mean / integral_born)          # normalisée à 2 car 2 Particules
    
    dist_real_all = np.array(dist_real_all)
    dist_ghost_all = np.array(dist_ghost_all)
    positions_p1_all = np.array(positions_p1_all)
    positions_p2_all = np.array(positions_p2_all)
    
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
        'dist_real': dist_real_all,
        'dist_ghost': dist_ghost_all,
        'positions_p1': positions_p1_all,
        'positions_p2': positions_p2_all
    }

# ===============================
# CALCUL DENSITÉS THÉORIQUES
# ===============================

def compute_theoretical_densities(x_space, phi1, phi2):
    """
    Calcule les densités marginales théoriques pour fermions et bosons.
    SANS sous-échantillonnage (grille complète).
    """
    dx_local = x_space[1] - x_space[0]
    Nx = len(x_space)
    
    # Normalisation des états individuels
    norm1 = np.sqrt(np.trapz(np.abs(phi1)**2, x_space))
    norm2 = np.sqrt(np.trapz(np.abs(phi2)**2, x_space))
    
    if norm1 > 0:
        phi1 = phi1 / norm1
    if norm2 > 0:
        phi2 = phi2 / norm2
    
    print(f"   Calcul sur grille {Nx}×{Nx}...")

    # Grilles 2D complètes
    phi1_x1 = phi1[:, np.newaxis]  # (Nx, 1)
    phi2_x2 = phi2[np.newaxis, :]  # (1, Nx)
    phi1_x2 = phi1[np.newaxis, :]
    phi2_x1 = phi2[:, np.newaxis]
    
    # Fonctions d'onde 2-particules
    psi_antisym = (phi1_x1 * phi2_x2 - phi1_x2 * phi2_x1) / np.sqrt(2)
    psi_sym = (phi1_x1 * phi2_x2 + phi1_x2 * phi2_x1) / np.sqrt(2)
    
    rho_2d_fermion = np.abs(psi_antisym)**2
    rho_2d_boson = np.abs(psi_sym)**2
    
    # Marginalisation
    rho1_fermion = np.trapz(rho_2d_fermion, x_space, axis=1)  # Intègre sur x₂
    rho2_fermion = np.trapz(rho_2d_fermion, x_space, axis=0)  # Intègre sur x₁
    
    rho1_boson = np.trapz(rho_2d_boson, x_space, axis=1)
    rho2_boson = np.trapz(rho_2d_boson, x_space, axis=0)
    
    # Normalisation finale
    rho1_fermion /= np.trapz(rho1_fermion, x_space)
    rho2_fermion /= np.trapz(rho2_fermion, x_space)
    rho1_boson /= np.trapz(rho1_boson, x_space)
    rho2_boson /= np.trapz(rho2_boson, x_space)
    
    # Densités totales théoriques
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
# COMPARAISON SCHRÖDINGER
# ===============================

def compare_schrodinger(x_space, sigma_target):
    """
    Évolution Schrödinger libre jusqu'à convergence.
    """
    dx_local = x_space[1] - x_space[0]
    psi_qm = np.exp(-0.5*(x_space/2.0)**2).astype(np.complex128)
    psi_qm /= np.sqrt(np.trapz(np.abs(psi_qm)**2, x_space))
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        # Laplacien
        lap_qm = np.zeros_like(psi_qm)
        lap_qm[1:-1] = (psi_qm[2:] - 2*psi_qm[1:-1] + psi_qm[:-2]) / dx_local**2
        
        # Évolution
        psi_qm += dt * (1j * omega * lap_qm)
        
        # Normalisation
        norm = np.sqrt(np.trapz(np.abs(psi_qm)**2, x_space))
        if norm > 0:
            psi_qm /= norm
        
        # Largeur actuelle
        rho_qm_temp = np.abs(psi_qm)**2
        rho_qm_temp /= np.trapz(rho_qm_temp, x_space)
        mean_x = np.trapz(x_space * rho_qm_temp, x_space)
        sigma_x = np.sqrt(np.trapz((x_space - mean_x)**2 * rho_qm_temp, x_space))
        
        if abs(sigma_x - sigma_target) < 0.01:
            print(f"   Convergence Schrödinger en {steps} steps")
            break
        
        steps += 1
    
    rho_qm = np.abs(psi_qm)**2
    rho_qm /= np.trapz(rho_qm, x_space)
 
    return rho_qm

# ===============================
# ANALYSE QUANTITATIVE
# ===============================

def analyze_results(data, theory):
    """
    Applique les tests du Code 1 : corrélation et erreur L¹.
    """
    x_space = data['x_space']
    
    # ========================================
    # 1. PARTICULE 1
    # ========================================
    # vs |ψ1|²
    corr_p1_born = np.corrcoef(data['rho_p1_obs'], data['born_p1'])[0, 1]
    error_L1_p1_born = 0.5 * np.trapz(np.abs(data['rho_p1_obs'] - data['born_p1']), x_space)
    
    # vs Théorie fermions
    corr_p1_fermion = np.corrcoef(data['rho_p1_obs'], theory['rho1_fermion'])[0, 1]
    error_L1_p1_fermion = 0.5 * np.trapz(np.abs(data['rho_p1_obs'] - theory['rho1_fermion']), x_space)
    
    # vs Théorie bosons
    corr_p1_boson = np.corrcoef(data['rho_p1_obs'], theory['rho1_boson'])[0, 1]
    error_L1_p1_boson = 0.5 * np.trapz(np.abs(data['rho_p1_obs'] - theory['rho1_boson']), x_space)
    
    # ========================================
    # 2. PARTICULE 2
    # ========================================
    corr_p2_born = np.corrcoef(data['rho_p2_obs'], data['born_p2'])[0, 1]
    error_L1_p2_born = 0.5 * np.trapz(np.abs(data['rho_p2_obs'] - data['born_p2']), x_space)
    
    corr_p2_fermion = np.corrcoef(data['rho_p2_obs'], theory['rho2_fermion'])[0, 1]
    error_L1_p2_fermion = 0.5 * np.trapz(np.abs(data['rho_p2_obs'] - theory['rho2_fermion']), x_space)
    
    corr_p2_boson = np.corrcoef(data['rho_p2_obs'], theory['rho2_boson'])[0, 1]
    error_L1_p2_boson = 0.5 * np.trapz(np.abs(data['rho_p2_obs'] - theory['rho2_boson']), x_space)
    
    # ========================================
    # 3. DENSITÉ TOTALE
    # ========================================
    corr_total_born = np.corrcoef(data['rho_total_obs'], data['born_sum'])[0, 1]
    error_L1_total_born = 0.5 * np.trapz(np.abs(data['rho_total_obs'] - data['born_sum']), x_space)
    
    corr_total_fermion = np.corrcoef(data['rho_total_obs'], theory['rho_total_fermion'])[0, 1]
    error_L1_total_fermion = 0.5 * np.trapz(np.abs(data['rho_total_obs'] - theory['rho_total_fermion']), x_space)
    
    corr_total_boson = np.corrcoef(data['rho_total_obs'], theory['rho_total_boson'])[0, 1]
    error_L1_total_boson = 0.5 * np.trapz(np.abs(data['rho_total_obs'] - theory['rho_total_boson']), x_space)
    
    # ========================================
    # 4. FONCTION g(r)
    # ========================================
    r_vals, g_r, hist_real, hist_ghost = compute_pair_correlation(
        data['dist_real'], data['dist_ghost'], x_min, x_max, bins=80
    )
    g_0 = g_r[np.argmin(np.abs(r_vals - 1.0))]
    
    # Facteur d'exclusion
    overlap_real = np.mean(data['dist_real'] < 2.0)
    overlap_ghost = np.mean(data['dist_ghost'] < 2.0)
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
    
    # Totale
    """.
    mean_x_total = np.trapz(x_space * data['born_sum'], x_space)
    sigma_x_total = np.sqrt(np.trapz((x_space - mean_x_total)**2 * data['born_sum'], x_space))
    rho_qm_total = compare_schrodinger(x_space, sigma_x_total, N_part=2)                                       
    """
    rho_qm_total = rho_qm_p1 + rho_qm_p2

    # Vérification symétrie
    print(f"\n5. LARGEURS :")
    print(f"   σ_P1    : {sigma_x_p1:.3f}")
    print(f"   σ_P2    : {sigma_x_p2:.3f}")
    # print(f"   σ_total : {sigma_x_total:.3f}")
    print(f"   Asymétrie : {abs(sigma_x_p1 - sigma_x_p2)/sigma_x_p1 * 100:.1f}%")

    # ========================================
    # AFFICHAGE
    # ========================================
    print("\n" + "="*70)
    print("RÉSULTATS QUANTITATIFS")
    print("="*70)
    
    print("\n1. PARTICULE 1 :")
    print(f"   vs |ψ₁|² observé  : corr={corr_p1_born:.4f}, L¹={error_L1_p1_born:.5f}")
    print(f"   vs Théorie fermion: corr={corr_p1_fermion:.4f}, L¹={error_L1_p1_fermion:.5f}")
    print(f"   vs Théorie boson  : corr={corr_p1_boson:.4f}, L¹={error_L1_p1_boson:.5f}")
    
    print("\n2. PARTICULE 2 :")
    print(f"   vs |ψ₂|² observé  : corr={corr_p2_born:.4f}, L¹={error_L1_p2_born:.5f}")
    print(f"   vs Théorie fermion: corr={corr_p2_fermion:.4f}, L¹={error_L1_p2_fermion:.5f}")
    print(f"   vs Théorie boson  : corr={corr_p2_boson:.4f}, L¹={error_L1_p2_boson:.5f}")
    
    print("\n3. DENSITÉ TOTALE :")
    print(f"   vs |ψ₁+ψ₂|² obs   : corr={corr_total_born:.4f}, L¹={error_L1_total_born:.5f}")
    print(f"   vs Théorie fermion: corr={corr_total_fermion:.4f}, L¹={error_L1_total_fermion:.5f}")
    print(f"   vs Théorie boson  : corr={corr_total_boson:.4f}, L¹={error_L1_total_boson:.5f}")
    
    print("\n4. CORRÉLATIONS DE PAIRES :")
    print(f"   g(r≈1)            : {g_0:.3f}")
    print(f"   Facteur d'exclusion: {exclusion_factor:.2f}x")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC")
    print("="*70)
    
    # Critère principal : convergence vers Born
    born_ok = (corr_p1_born > 0.95 and corr_p2_born > 0.95 and 
               error_L1_p1_born < 0.05 and error_L1_p2_born < 0.05)
    
    if born_ok and g_0 < 0.5 and exclusion_factor > 2.0:
        print("🎉 COMPORTEMENT FERMIONIQUE CONFIRMÉ")
        print(f"   ✓ Convergence Born : ρ ≈ |ψ|² (corr>{0.95:.2f})")
        print(f"   ✓ Meilleur avec théorie fermions")
        print(f"   ✓ Trou de Fermi : g(1)={g_0:.3f} < 0.5")
    elif born_ok:
        print("✓ CONVERGENCE BORN VALIDÉE")
        print("⚠️  Signature fermionique partielle")
    else:
        print("❌ Convergence Born insuffisante")
    
    return {
        'corr_p1_born': corr_p1_born,
        'error_L1_p1_born': error_L1_p1_born,
        'corr_p1_fermion': corr_p1_fermion,
        'error_L1_p1_fermion': error_L1_p1_fermion,
        'corr_total_born': corr_total_born,
        'error_L1_total_born': error_L1_total_born,
        'corr_total_fermion': corr_total_fermion,
        'error_L1_total_fermion': error_L1_total_fermion,
        'g_r': (r_vals, g_r),
        'g_0': g_0,
        'exclusion_factor': exclusion_factor,
        'hist_real': hist_real,
        'hist_ghost': hist_ghost,
        'rho_qm_p1': rho_qm_p1,      
        'rho_qm_p2': rho_qm_p2,      
        'rho_qm_total': rho_qm_total        
    }

# ===============================
# VISUALISATION (PAGE 1)
# ===============================

def plot_results_page1(data, theory, metrics):
    """
    Page 1 : Densités individuelles + totale + résidus
    """
    x = data['x_space']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ========================================
    # 1. PARTICULE 1 - Densités
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    mean_pos_p1 = np.average(x, weights=data['rho_p1_obs'])
    ax1.plot(x, data['rho_p1_obs'], 'b-', lw=2.5, label='ρ₁ observée')
    ax1.plot(x, data['born_p1'], 'k--', lw=2, alpha=0.8, label='|ψ₁|² (Born)')
    ax1.plot(x, theory['rho1_fermion'], 'r:', lw=2, label='Théorie fermion')
    ax1.plot(x+mean_pos_p1, theory['rho1_boson'], 'g:', lw=1.5, alpha=0.6, label='Théorie boson')
    ax1.plot(x+mean_pos_p1, metrics['rho_qm_p1'], 'm:', lw=1.5, alpha=0.6, label='Schrödinger')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Densité')
    ax1.set_title(f'Particule 1 (corr_Born={metrics["corr_p1_born"]:.4f}, L¹={metrics["error_L1_p1_born"]:.5f})', 
                 fontweight='bold', fontsize=11)
    
    # ========================================
    # 2. PARTICULE 2 - Densités
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    mean_pos_p2 = np.average(x, weights=data['rho_p2_obs'])
    ax2.plot(x, data['rho_p2_obs'], 'b-', lw=2.5, label='ρ₂ observée')
    ax2.plot(x, data['born_p2'], 'k--', lw=2, alpha=0.8, label='|ψ₂|² (Born)')
    ax2.plot(x, theory['rho2_fermion'], 'r:', lw=2, label='Théorie fermion')
    ax2.plot(x+mean_pos_p2, theory['rho2_boson'], 'g:', lw=1.5, alpha=0.6, label='Théorie boson')
    ax2.plot(x+mean_pos_p2, metrics['rho_qm_p2'], 'm:', lw=1.5, alpha=0.6, label='Schrödinger')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Densité')
    ax2.set_title('Particule 2', fontweight='bold', fontsize=11)
    
    # ========================================
    # 3. RÉSIDUS P1
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
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Résidus')
    ax3.set_title('Résidus P1', fontweight='bold', fontsize=11)
    
    # ========================================
    # 4. RÉSIDUS P2
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
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('Résidus')
    ax4.set_title('Résidus P2', fontweight='bold', fontsize=11)
    
    # ========================================
    # 5. DENSITÉ TOTALE
    # ========================================
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(x, data['rho_total_obs'], 'b-', lw=2.5, label='ρ₁+ρ₂ observée')
    ax5.plot(x, data['born_sum'], 'k--', lw=2, alpha=0.8, label='|ψ₁+ψ₂|² (Born)')
    ax5.plot(x, theory['rho_total_fermion'], 'r:', lw=2, label='Théorie fermion')
    ax5.plot(x, theory['rho_total_boson'], 'g:', lw=1.5, alpha=0.6, label='Théorie boson')
    ax5.plot(x, metrics['rho_qm_total'], 'm:', lw=1.5, alpha=0.6, label='Schrödinger')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.set_xlabel('Position x', fontsize=11)
    ax5.set_ylabel('Densité totale', fontsize=11)
    ax5.set_title(f'Densité totale (corr_Born={metrics["corr_total_born"]:.4f}, L¹={metrics["error_L1_total_born"]:.5f})', 
                 fontweight='bold', fontsize=12)
    
    plt.suptitle('Page 1 : Convergence Born et Comparaison Théories', 
                fontsize=14, fontweight='bold', y=0.995)
    
    base_name = "Pauli_v4_Page1"
    i = 1
    while os.path.exists(f"{base_name}_{i}.png"):
        i += 1
    filename = f"{base_name}_{i}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Page 1 sauvegardée : {filename}")

# ===============================
# VISUALISATION (PAGE 2)
# ===============================

def plot_results_page2(data, theory, metrics):
    """
    Page 2 : g(r), distances, heatmap 2D
    """
    x = data['x_space']
    r_vals, g_r = metrics['g_r']
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================
    # 1. FONCTION g(r)
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r_vals, g_r, 'r-', linewidth=2.5, label='g(r) mesuré')
    ax1.axhline(1, color='gray', linestyle='--', linewidth=1.5, label='Particules indépendantes')
    ax1.fill_between(r_vals, g_r, 1, where=(g_r < 1), 
                     facecolor='red', alpha=0.2, label='Trou de Fermi')
    ax1.set_xlabel('Distance r', fontsize=11)
    ax1.set_ylabel('g(r)', fontsize=11)
    ax1.set_title(f'Corrélation de paires (g(1)={metrics["g_0"]:.3f})', 
                 fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 2)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # ========================================
    # 2. DISTRIBUTION DES DISTANCES (CORRIGÉE)
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    # Plot direct des histogrammes (pas refaire un hist d'un hist !)
    ax2.plot(r_vals, metrics['hist_ghost'], 'gray', lw=2, alpha=0.6, label='Fantômes')
    ax2.fill_between(r_vals, metrics['hist_ghost'], alpha=0.3, color='gray')
    ax2.plot(r_vals, metrics['hist_real'], 'r-', lw=2.5, label='Réel')
    ax2.fill_between(r_vals, metrics['hist_real'], alpha=0.2, color='red')
    ax2.set_xlabel('Distance |x₁ - x₂|', fontsize=11)
    ax2.set_ylabel('Densité de probabilité', fontsize=11)
    ax2.set_title(f'Distribution distances (Excl: {metrics["exclusion_factor"]:.2f}x)', 
                 fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 40)
    
    # ========================================
    # 3. HEATMAP 2D : Position(P1) vs Position(P2)
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    H, xedges, yedges = np.histogram2d(data['positions_p1'], data['positions_p2'], 
                                        bins=60, range=[[x_min, x_max], [x_min, x_max]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(H.T, origin='lower', extent=extent, cmap='hot', aspect='auto', 
                    interpolation='bilinear')
    
    # Diagonale (positions identiques - interdit pour fermions)
    ax3.plot([x_min, x_max], [x_min, x_max], 'cyan', linewidth=2.5, 
             linestyle='--', label='x₁=x₂ (interdit)')
    
    ax3.set_xlabel('Position P1', fontsize=11)
    ax3.set_ylabel('Position P2', fontsize=11)
    ax3.set_title('Heatmap positions jointes', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=10, loc='upper left')
    plt.colorbar(im, ax=ax3, label='Densité')
    
    # ========================================
    # 4. HEATMAP THÉORIQUE FERMIONS
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    rho_2d = theory['rho_2d_fermion']
    extent_theory = [x_min, x_max, x_min, x_max]
    im2 = ax4.imshow(rho_2d.T, origin='lower', extent=extent_theory, cmap='hot', 
                     aspect='auto', interpolation='bilinear')
    
    # Diagonale théorique
    ax4.plot([x_min, x_max], [x_min, x_max], 'cyan', linewidth=2.5, 
             linestyle='--', label='x₁=x₂')
    
    ax4.set_xlabel('Position P1', fontsize=11)
    ax4.set_ylabel('Position P2', fontsize=11)
    ax4.set_title('Théorie fermions |ψ₋(x₁,x₂)|²', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=10, loc='upper left')
    plt.colorbar(im2, ax=ax4, label='Densité théorique')
    
    plt.suptitle('Page 2 : Corrélations spatiales et Exclusion de Pauli', 
                fontsize=14, fontweight='bold', y=0.995)
    
    base_name = "Pauli_v4_Page2"
    i = 1
    while os.path.exists(f"{base_name}_{i}.png"):
        i += 1
    filename = f"{base_name}_{i}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Page 2 sauvegardée : {filename}")

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    # 1. Simulation
    data = run_pauli_simulation()
    
    # 2. Calcul des densités théoriques
    print("\n🔬 Calcul des densités théoriques...")
    theory = compute_theoretical_densities(
        data['x_space'], data['phi1_mean'], data['phi2_mean']
    )
    
    # 3. Analyse quantitative
    metrics = analyze_results(data, theory)
    
    # 4. Visualisation (2 pages)
    plot_results_page1(data, theory, metrics)
    plot_results_page2(data, theory, metrics)
    
    print("\n" + "="*70)
    print("✓ SIMULATION TERMINÉE")
    print("="*70)
