# Born Rule & Polarization in 2D
High-performance implementation using Taichi (GPU parallelization).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zobeewan/Born_Rule_emergence-Pilot_Wave/blob/main/notebooks/simulation_2D_Born_&_polarization_GPU_Colab.ipynb)   
**⏱ Runtime:** ~20-100 minutes (depending on params)    

The first cell is optional, use it if you want to save your results in your Google Drive

Increase N_TOTAL_RUNS for better statistical convergence. 

But for Google Colab free use only, 10,000 is usually near the free daily limit

Reduce it to shorten simulation time, every 1,000 runs ≈ 10min (minimum step is bound to 500 ≈ 5min)

To simulate less than 500, up to a single particle, use `src/simulation_2D_Born_&_polarization.py` instead
