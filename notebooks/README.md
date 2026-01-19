# Born Rule & Polarization in 2D (GPU)
High-performance 2D pilot-wave simulation using **Taichi (GPU parallelization)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zobeewan/Born_Rule_emergence-Pilot_Wave/blob/main/notebooks/simulation_2D_Born_&_polarization_GPU_Colab.ipynb)   
**⏱ Runtime:** ~20-100 minutes (depending on params)    

---

## ▶️ How to run

- Open the notebook in **Google Colab**
- (Optional) Run the **first cell** if you want to save results to your Google Drive
- Run all cells sequentially

---

## ⏱ Key Scaling Parameter

**N_TOTAL_RUNS** controls the number of individual particles simulate
(minimum step is bound to **300 runs** (~ 1 min 15s))
- ⚠️ Increasing `N_TOTAL_RUNS` improves convergence but increases runtime linearly
- Recommended for Colab (free): **~24,000 runs**
- Higher values may hit the **daily usage limit**
- ~ 5 minutes per **1,200 runs**

---

## 🖥 CPU Alternative

For small runs (≤500 particles), up to single-particle tests, use the CPU version instead:
```text
src/simulation_2D_Born_&_polarization.py
```
