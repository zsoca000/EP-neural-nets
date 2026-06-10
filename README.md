# The Risk of Method-Specific Overfitting: Neural Network Surrogates of Elastic Plastic Behavior

This repository is the official code companion for the paper: **"The Risk of Method-Specific Overfitting: Neural Network Surrogates of Elastic Plastic Behavior"**.

It contains the implementation of the data generators, 1D elastoplastic constitutive models, surrogate neural network architectures, hyperparameter sweeps, and Jupyter notebooks used for evaluation and figure generation to identify and analyze the phenomenon of **method-specific overfitting** in path-dependent sequence modeling.

---

## Repository Structure

```text
EP-neural-nets/
│
├── configs/                 # Configuration and material parameters
│   ├── generators.yaml      # Settings for the loading history generators
│   ├── materials.py         # Material parameters for the constitutive models
│   └── train.yaml           # Training and early stopping hyperparameters
│
├── data/                    # Data pipeline (generated inputs/outputs/states ignored by git)
│   ├── data_set.py          # Lazy-loading wrapper casting dataset binaries to PyTorch tensors
│   ├── generators.py        # Implementation of stochastic and critical loading history generators
│   └── materials.py         # Radial return numerical solvers for the 1D constitutive models
│
├── models/                  # Neural network surrogate engines
│   ├── models.py            # Autoregressive rollout simulations and preprocessor scaling
│   ├── networks.py          # PyTorch core MLP and LSTM architectures
│   └── utils.py             # MinMaxScaler and ErrorMetrics calculation helpers
│
├── notebooks/               # Jupyter notebooks for evaluation and figure generation
│   ├── 00_data_visualization.ipynb
│   ├── 01_grid_search.ipynb
│   ├── 02_comparison.ipynb
│   └── 03_visualize_responses.ipynb
│
├── utils/                   # Non-model formatting and timing helpers
│   ├── animations.py        # Compiles image frames into training progress videos
│   ├── readable.py          # LaTeX formatter for plot labels and dataset strings
│   └── time.py              # Time conversion utility
│
├── eval.py                  # Evaluator engine for trained models
├── train.py                 # PyTorch training pipeline wrapper
└── tasks.py                 # Main entry point to run data generation, sweeps, and evaluations
```

---

## Covered Scope

### 1D Elastoplastic Constitutive Models
We cover six classical hardening laws ($m_1$ to $m_6$) implemented via custom radial return algorithms:
* **Isotropic hardening**: Linear ($m_1$), Swift non-linear ($m_2$)
* **Kinematic hardening**: Linear/Prager ($m_3$), Non-linear Armstrong–Frederick ($m_4$)
* **Mixed hardening**: Linear + Linear ($m_5$), Linear + Armstrong–Frederick non-linear ($m_6$)

### Loading History Generators
* **Stochastic (Training/Eval)**: Random Walk (RW), Gaussian Process (GP), Power Decay Multisine (PD-MS), Baseline Multisine (BL-MS)
* **Critical (Adversarial Testing)**: Impulse-like, Increasing Amplitude (Extrapolation), Cyclic (Numerical Drift), Resolution (Temporal Discretization), Piecewise Linear

### Surrogate Neural Network Families
* **AR-MLP / Res-AR-MLP**: Autoregressive Multilayer Perceptrons modeling absolute stress values or stress increments.
* **LSTM / Res-LSTM**: Long Short-Term Memory networks modeling absolute stress values or stress increments.

---

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zsoca000/EP-neural-nets.git
cd EP-neural-nets
pip install -r requirements.txt
```

### 2. Execution Workflow
You can run the entire workflow step-by-step using Python scripts or by executing the numbered Jupyter notebooks in the `notebooks/` directory.

#### Step 1: Generate Loading Histories and Material Responses
Generates the stochastic and critical datasets:
```bash
python -c "from tasks import generate_data; generate_data()"
```

#### Step 2: Execute Hyperparameter Grid Search
Sweeps across 184 configurations (AR-MLP, LSTM) on the Swift isotropic material ($m_2$) using datasets generated via the Power Decay Multisine (PD-MS) algorithm:
```bash
python -c "from tasks import run_grid_search; run_grid_search()"
```

#### Step 3: Train Optimized Surrogate Across All Materials and Generators
Fits the optimized surrogate architecture (AR-MLP with parameters $k=3, p=5, q=3$ in incremental mode) across all combinations of the six reference constitutive models and four stochastic loading history generators:
```bash
python -c "from tasks import run_cross_material_training; run_cross_material_training()"
```

#### Step 4: Evaluate Robustness Metrics
Computes cross-generator evaluation metrics and exports them to JSON:
```bash
python -c "from tasks import eval_all; eval_all()"
```

---

## Citation
If you use this code or our methodology in your research, please cite our paper:
```bibtex
@article{szasz2026overfitting,
  title={The Risk of Method-Specific Overfitting: Neural Network Surrogates of Elastic Plastic Behavior},
  author={Szasz, Zsolt and Kossa, Attila},
  journal={},
  year={2026}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
