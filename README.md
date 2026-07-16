# Single-cell RNA-to-Protein Prediction

This project predicts surface protein abundance from single-cell RNA
expression using the CITE-seq dataset from the NeurIPS 2022 Open Problems
in Multimodal Learning competition.

The primary objective is to evaluate whether transcriptional information
from observed time points can generalize to an unseen time point.

## Problem

The CITE-seq dataset contains paired measurements of:

- RNA expression: 22,050 gene features per cell
- Surface protein abundance: 140 protein targets per cell
- Metadata: collection day, donor and cell type information

A total of 70,988 aligned cells were used. Models were trained on Day 2
and Day 3 cells and evaluated on Day 4 cells.

This time-based split is more challenging than a random split because it
tests temporal generalization to an unseen stage of cell differentiation.

## Data split

| Split | Days | Number of cells |
|---|---|---:|
| Training | Day 2 and Day 3 | 42,843 |
| Temporal validation | Day 4 | 28,145 |

Day 4 was excluded from model fitting and PCA estimation.

## Feature engineering

RNA expression is high-dimensional, sparse and noisy. PCA was used to
reduce 22,050 RNA features to 50 principal components.

To avoid information leakage:

1. PCA was fitted using only Day 2–3 cells.
2. The fitted transformation was applied to Day 4 cells.
3. Model hyperparameters and scalers were fitted without using Day 4
   protein targets.

The final model comparison used the same 50 PCA features for Ridge and
the multi-output MLP.

## Models

### Mean-profile baseline

The mean protein profile of the Day 2–3 training cells was assigned to
every Day 4 cell. This is the optimal constant predictor under squared
error loss.

### Ridge regression

A multi-output Ridge regression model was fitted using standardized PCA
features. Ridge provides a computationally efficient and interpretable
linear baseline.

### Multi-output MLP

A neural network with architecture 50–64–32–140 was trained to predict
all 140 proteins simultaneously. The model uses ReLU activations and
dropout regularization.

An internal split of the Day 2–3 data was used to monitor training. The
held-out Day 4 data was used only for temporal evaluation.

## Evaluation metrics

The primary metric is the competition-style mean row-wise Pearson
correlation. Pearson correlation is calculated across the 140 proteins
for each cell and then averaged across cells.

Two secondary metrics are also reported:

- Mean and median protein-wise Pearson correlation
- Root mean squared error (RMSE)

## Results

| Model | Row-wise Pearson | Mean protein PCC | Median protein PCC | RMSE |
|---|---:|---:|---:|---:|
| Mean-profile | 0.7776 | Undefined | Undefined | 2.2916 |
| Ridge | 0.8755 | 0.4088 | 0.3922 | 1.6815 |
| Multi-output MLP | 0.8727 | 0.4084 | 0.3948 | 1.7076 |

On the Day 4 temporal validation set, Ridge achieved the best overall
performance, with a mean row-wise Pearson correlation of **0.8755**,
slightly outperforming the multi-output MLP (**0.8727**).

The two models achieved nearly identical mean protein-wise PCC values
(Ridge: **0.4088**; MLP: **0.4084**), while Ridge also obtained a lower
RMSE (Ridge: **1.6815**; MLP: **1.7076**).

## Repository structure

- `configs/`: experiment configuration
- `src/scprotein/`: reusable data, feature, model and evaluation functions
- `scripts/`: executable preprocessing, training and plotting workflows
- `notebooks/`: final reader-facing analysis notebook
- `results/`: metrics, coefficient tables and figures
- `report/`: final written report

Raw Kaggle data are not redistributed with this repository.

## Installation

Create and activate a Python environment:

```bash
conda create -n single_cell_prediction python=3.11
conda activate single_cell_prediction
pip install -r requirements.txt