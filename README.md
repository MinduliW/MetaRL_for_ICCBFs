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
- **Empirically safety guarantees** via ICCBFs  
- **Long-horizon performance awareness** via meta-learned class-𝒦 parameters  
- **Recoverability beyond conservative invariant sets** using learned residual barriers
- - **Robustness to hidden parameters and noise** using recurrent neural networks

The result is **non-greedy, fuel-efficient, and safe autonomy** for realistic mission profiles.

---

## 🎥 Demos (Real rollouts)

### Cruise Control (Input-constrained safety)
![Demo](media/CCGif.gif)


---

### Docking & Final Approach
![Demo](media/docking.gif)


---

### Autonomous Inspection (KOZ, KIZ, Sun constraints)

This is my favourite test case. The first result has no ICCBF tunning, and acheives the inspection score very fast, but at a high fuel cost. The second one shows ICCBF tunning with MLP, which is not very successful. It lowers fuel consumption at the cost of inspection score. The last one is where ICCBF is tunned by RNN, here, the RNN is able to both lower the fuel consumption while still satisfying full inspection 75% of the time. The inspection takes longer of course compared to the first case, but it reaches completion. This defines a nongreedy control barrier system.

![Demo](media/inspectionuRL.gif)
*Figure: ICCBF inspection trajectories where only the nominal control is determined by RL. No ICCBF tuning. Cyan indicates task completion; blue indicates in-progress inspection.*



![Demo](media/inspectionNN.gif)
*Figure: NN-tuned ICCBF inspection trajectories. Cyan indicates task completion; blue indicates in-progress inspection.*


![Demo](media/inspectionRNN.gif)
*Figure: RNN-tuned ICCBF inspection trajectories. Cyan indicates task completion; blue indicates in-progress inspection.*




---

## 🧠 Key Contributions

- **Meta-RL tuning of ICCBF decay parameters**
  - Learns state-dependent class-𝒦 functions

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
