
# STAR RKHS Weightings

Code for the paper *Shapley Values of Structured Additive Regression Models and Application to RKHS Weightings of Functions* (Transactions in Machine Learning Research, 2025). 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gadub44/star-rkhs-weightings
   ```
2. Navigate to the project directory:
   ```bash
   cd star-rkhs-weightings
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## <a name="reproducing"></a>Reproducing the experiments 

The paper's experiments, including figures and tables, can be reproduced by running the following commands.

Figure 1:
```bash
python -m major_experiments.shapley_time
```
Figure 2:
```bash
python -m major_experiments.shapley_accuracy
```
Table 6:
```bash
python -m major_experiments.regression --final
```
Table 8:
```bash
python -m major_experiments.time_series --final
```
Figures 3 and 4:
```bash
python -m major_experiments.shapley_comparison
```

## <a name="starshap"></a>Using STAR-SHAP
This repository contains STAR-SHAP, the Shapley value algorithm introduced in the paper. The explainer class is Shapley.STAR_Explainer. Given a model and data, the explainer can be used as follows:

```
from Shapley import STAR_Explainer
explainer = STAR_Explainer(model, X)
values = explainer.shap_values(X)
```

Importantly, the model must be a Structured Additive Regression model and implement the following functions:
```
model.partial(fs) # Get the partial model corresponding to the feature subset fs.
model.get_all_unique_fs() # Get all feature subsets used by the model.
```
The model must also be callable, i.e. returns an output when given an instance:
```
model(x) # Get the output of the model on instance x.
```

## Using RKHS Weightings

todo