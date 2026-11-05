# CEAM — Cross-Effect Adaptive Model

**Reference implementation of a hybrid neural discrete-choice model**

CEAM combines an explicit linear utility with a neural correction conditioned on the attributes and baseline utilities of all alternatives:

```math
P(i\mid x)=\frac{\exp(V_i(x)+\delta_i(x,V(x)))}{\sum_j\exp(V_j(x)+\delta_j(x,V(x)))}
```

## Implementation

- **Utility layer:** shared time and cost coefficients, alternative-specific constants (Train is the reference).
- **Neural correction:** one fully connected hidden layer (ReLU), receiving six scaled SwissMetro attributes and three linear utilities; three correction outputs.
- **Prediction:** availability-masked softmax of linear utilities plus neural corrections.
- **Training:** Adam with a batch-mean cross-entropy loss and a **batch-summed** squared-L2 penalty on neural correction outputs. The penalty is evaluated over each minibatch; its effective contribution depends on batch size. Example hyperparameters are configurable.
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

The input must contain the six **pre-scaled** fields `TRAIN_TT_SCALED`, `TRAIN_COST_SCALED`, `SM_TT_SCALED`, `SM_COST_SCALED`, `CAR_TT_SCALED`, `CAR_CO_SCALED`; availability flags `TRAIN_AV`, `SM_AV`, `CAR_AV`; and `PURPOSE`, `CHOICE`, `ID`. The examples default to respondent-disjoint train/validation/test splits.

## Scope

This public repository contains the **core model**, training and optimization utilities, an independent MNL baseline, basic response diagnostics, automated tests, and illustrative SwissMetro command-line examples. It focuses on the reusable modeling framework and runnable examples. Study-specific datasets, preprocessing pipelines, experiment configurations, and supplementary analysis workflows are outside the scope of this release. 

## Citation

*Cross-Effect Adaptive Model: Integrating Neural Cross-Interactions into Utility-Based Discrete Choice Models.* Publication details to follow.

## License

MIT — see [LICENSE](LICENSE).
