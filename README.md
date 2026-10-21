# Human-Motion-Prediction
# 3D Human Motion Prediction on Human3.6M

## Overview

This repository provides the implementation of a coarse-to-fine 3D human motion prediction framework combining **Transformer-based motion forecasting** and **LightGBM-based residual refinement**.

The framework first employs a Transformer to capture temporal dependencies and generate coarse predictions of future 3D human poses. LightGBM is then used to learn and correct prediction residuals, improving the accuracy of short-term human motion prediction.

Experiments are conducted on the **Human3.6M** benchmark dataset.

## Key Features

- **Transformer-based Motion Prediction:** Captures temporal dependencies in human motion sequences.
- **LightGBM Residual Refinement:** Improves prediction accuracy through structured residual learning.
- **Coarse-to-Fine Framework:** Combines deep learning and gradient boosting for accurate motion forecasting.
- **Human3.6M Benchmark:** Supports training, prediction, and evaluation on the Human3.6M dataset.

## Repository Contents

The repository includes Python scripts for:

- Data preprocessing and loading
- Transformer model training and prediction
- LightGBM residual training and refinement
- Motion prediction and performance evaluation

## Dataset

The experiments are conducted on **Human3.6M**, a widely used benchmark for 3D human motion prediction.

The dataset is not included in this repository and must be obtained separately.

## Requirements

- Python
- PyTorch
- LightGBM
- NumPy
- SciPy
- Other dependencies required by the implementation

## Acknowledgments

Parts of the code in this repository are adapted from the open-source implementation of **AuxFormer**. We sincerely thank the authors of AuxFormer for making their code publicly available and acknowledge their valuable contributions to this research.

## License

Please respect the licenses and attribution requirements of the original open-source implementations used in this project.
