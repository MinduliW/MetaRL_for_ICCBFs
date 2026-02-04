# Meta-Reinforcement Learning for Input-Constrained Control Barrier Functions (ICCBFs)

<p align="center">
  <b>Robust & fuel-efficient autonomy for spacecraft proximity operations</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Spacecraft%20RPO-blue">
  <img src="https://img.shields.io/badge/Safety-ICCBFs-green">
  <img src="https://img.shields.io/badge/Learning-Meta--RL-orange">
</p>

---

## 🚀 Overview

This repository contains the reference implementation for **Meta-Reinforcement Learning–aided Input-Constrained Control Barrier Functions (ICCBFs)** applied to **safety-critical spacecraft proximity operations**, including:

- Cruise control (input-limited safety)
- Docking and final approach
- On-orbit inspection with keep-out, keep-in, and illumination constraints

The framework combines:
- **Provable safety guarantees** via ICCBFs  
- **Long-horizon performance awareness** via meta-learned class-𝒦 parameters  
- **Recoverability beyond conservative invariant sets** using learned residual barriers  

The result is **non-greedy, fuel-efficient, and safe autonomy** for realistic mission profiles.

---

## 🎥 Demos (Real rollouts)

### Cruise Control (Input-constrained safety)
https://github.com/MinduliW/MetaRL_for_ICCBFs/raw/main/media/cruise_control_RNN.mp4

---

### Docking & Final Approach
https://github.com/MinduliW/MetaRL_for_ICCBFs/raw/main/media/docking.mp4

---

### Autonomous Inspection (KOZ, KIZ, Sun constraints)
https://github.com/MinduliW/MetaRL_for_ICCBFs/raw/main/media/inspection.mp4

---

## 🧠 Key Contributions

- **Meta-RL tuning of ICCBF decay parameters**
  - Learns state-dependent class-𝒦 functions
  - Preserves forward invariance of certified safe sets

- **Recoverability beyond conservative invariant sets**
  - Enables safe task completion from a subset of traditionally abandoned states

- **Fuel-aware autonomy**
  - Orders-of-magnitude reduction in total Δv for inspection and docking tasks

- **Inspection-aware safety**
  - Simultaneous handling of KOZ, KIZ, and Sun-angle constraints

---

## 📊 Representative Results

- Strong reduction in median and tail Δv consumption
- Near-perfect inspection completion rates for learned ICCBFs
- Improved robustness compared to untuned ICCBF and naive RL baselines

(See `src/inspection/plot_inspection_results_with_violin.py` for evaluation scripts.)

---

## 🗂 Repository Structure

```text
src/
 ├─ data/              # All .mat files and initial state sets stored here
 ├─ cruise_control/     # Cruise control experiments
 ├─ docking/            # Docking & final approach
 └─ inspection/         # Inspection task + evaluation
Notebooks/
 ├─ TrainedModels  # stores trainedmodels from the notebooks
 ├─ cruise_control.ipynb  # cruise control problem 
 ├─ docking.ipynb  #docking problem 
 ├─ inspection.ipynb #inspection problem
 ├─ images #plots used in the jupyter notebooks
media/
 ├─ cruise_control.mp4
 ├─ docking.mp4
 └─ inspection.mp4
