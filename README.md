# I2LDL
Code for our paper titled Tail-Aware Reconstruction of Incomplete Label Distributions with Low-Rank and Sparse Modeling.

## Requirements
To run this code, you need:

MATLAB (R2018a or newer recommended)

Optimization Toolbox™ (for quadprog)

Control System Toolbox™ (for sylvester)

## Inputs:
features: An n x d matrix of instance features.

obrT: An n x m binary matrix indicating which labels are observed.

labels: An n x m matrix of ground-truth labels.

lambda_1, lambda_2, lambda_3: Regularization parameters.

rho_1, rho_2: ADMM penalty parameters.

## Outputs:
W: A d x m weight matrix for the low-rank component.

Q: A d x m weight matrix for the sparse component.
