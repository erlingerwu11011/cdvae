# Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data.

The implementation utilizes the **Hydra** package for configuration management and **PyTorch Lightning** for model building and training. The code structure follows best practices from the [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template).

Baseline implementations, including **RMSN, CRN, G-Net, and Causal Transformer**, extended from the [Causal Transformer](https://arxiv.org/abs/2204.07258), originally available [here](https://github.com/Valentyn1997/CausalTransformer) under the MIT license. **Causal CPC** is adapted from [this paper](https://arxiv.org/pdf/2406.00535).

---

## Installation
To set up the environment and install dependencies, run:
```sh
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

---

## Configuration
- **Model configurations**: Located in `settings/model/`
- **Dataset configurations**: Located in `settings/dataset/`

For **MIMIC-III data**, place the file [all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract) in:
```
data/processed/
```

---

## Usage
To train CDVAE with a specific seed:
```sh
PYTHONPATH=. python3 runnables/train_cdvae.py exp.seed=10
```
