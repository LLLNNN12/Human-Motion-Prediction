# Human-Motion-Prediction
# 3D Human Motion Prediction on Human3.6M

## Overview

This repository provides the implementation of a coarse-to-fine 3D human motion prediction framework based on **Transformer and LightGBM**, evaluated on the **Human3.6M** dataset.

The framework employs a Transformer to capture temporal dependencies and generate coarse motion predictions, followed by LightGBM-based residual refinement to improve prediction accuracy.

## Key Features

- **Transformer-based Prediction:** Models temporal dependencies in human motion sequences to predict future 3D poses.
- **LightGBM Residual Refinement:** Refines coarse predictions by learning structured prediction residuals.
- **Human3.6M Benchmark:** Supports model training and evaluation on the Human3.6M dataset.
- **Short-term Motion Prediction:** Focuses on accurate future human pose prediction over short time horizons.

## Dataset

Experiments are conducted on the **Human3.6M** dataset, a widely used benchmark for 3D human pose estimation and motion prediction.

The dataset is not included in this repository and must be obtained separately.

## Requirements

- Python
- PyTorch
- LightGBM
- NumPy
- SciPy

## Usage

The provided code supports model training and evaluation on Human3.6M. Please configure the dataset paths and relevant parameters according to your local environment.

## Acknowledgment

This implementation is provided for academic research and reproducibility in 3D human motion prediction.
