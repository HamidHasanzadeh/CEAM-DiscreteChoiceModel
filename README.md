# CEAM — Cross-Effect Adaptive Model

**Reference implementation of a hybrid neural discrete-choice model**

CEAM combines an explicit linear utility with a neural correction conditioned on the attributes and baseline utilities of all alternatives:

```math
P(i\mid x)=\frac{\exp(V_i(x)+\delta_i(x,V(x)))}{\sum_j\exp(V_j(x)+\delta_j(x,V(x)))}
```

## Implementation

- **Utility layer:** shared time and cost coefficients, alternative-specific constants (Train is the reference).
- **Neural correction:** configurable fully connected hidden layers (ReLU, Tanh or GELU), receiving six scaled SwissMetro attributes and three linear utilities and producing three correction outputs. The default is one ReLU hidden layer.
- **Prediction:** availability-masked softmax of linear utilities plus neural corrections.
- **Training:** Adam with mean cross-entropy and a batch-normalized squared-L2 penalty on neural correction outputs. A batch-summed reduction remains available for compatibility.
- **Tuning:** optional CMA-ES hyperparameter search with validation log-loss.

## Run

Python 3.10+:

```bash
git clone https://github.com/HamidHasanzadeh/CEAM-DiscreteChoiceModel.git
cd CEAM-DiscreteChoiceModel
python -m pip install -e ".[test]"
python -m examples.train_ceam --data /path/to/swissmetro_prepared.csv
python -m examples.train_mnl --data /path/to/swissmetro_prepared.csv
python -m pytest
```

For a custom neural architecture, add `--hidden-layers 32,16 --activation gelu`. Use `--penalty-reduction sum` when the batch-summed objective is required. These are configuration options for the reference implementation, not reported study settings.

The input must contain the six **pre-scaled** fields `TRAIN_TT_SCALED`, `TRAIN_COST_SCALED`, `SM_TT_SCALED`, `SM_COST_SCALED`, `CAR_TT_SCALED`, `CAR_CO_SCALED`; availability flags `TRAIN_AV`, `SM_AV`, `CAR_AV`; and `PURPOSE`, `CHOICE`, `ID`. The examples default to respondent-disjoint train/validation/test splits.

## Repository Overview

This repository provides a streamlined implementation of CEAM, including the model architecture, training and hyperparameter optimization, an independent MNL baseline, diagnostic utilities, and automated tests. The SwissMetro examples demonstrate how to train and evaluate the models using appropriately prepared input data.

## Citation

*Cross-Effect Adaptive Model: Integrating Neural Cross-Interactions into Utility-Based Discrete Choice Models.* Publication details to follow.

## License

MIT — see [LICENSE](LICENSE).
