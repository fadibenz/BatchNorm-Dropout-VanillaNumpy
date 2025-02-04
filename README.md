# Batch Normalization and Dropout Implementation in Vanilla NumPy

## Overview
This repository contains an implementation of Batch Normalization and Dropout layers in vanilla NumPy, along with experiments demonstrating their effects on training deep neural networks. The implementations are integrated into fully connected networks and tested on the CIFAR-10 dataset. This work uses starter code from UC Berkeley's CS182 course.

## Key Features
- **Batch Normalization**: Implementation of forward and backward passes for batch normalization, including train/test mode handling.
- **Dropout**: Implementation of dropout regularization with forward and backward passes.
- **Deep Networks**: Integration of these layers into fully connected networks for image classification.
- **Experiments**: Comparative studies between networks with/without batch normalization and different dropout rates.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- PyTorch (for random seed setup)
- CIFAR-10 dataset

Install dependencies with:
```bash
pip install numpy matplotlib torch
```

## Implementation Details

### Batch Normalization
- **Forward Pass**: Normalizes activations using mini-batch statistics during training, and running averages during inference.
- **Backward Pass**: Correctly implements gradient computation through normalization operations.
- **Network Integration**: Added to fully connected layers with learnable parameters `gamma` and `beta`.

### Dropout
- **Forward Pass**: Randomly zeroes activations with probability `p` during training, scales activations during inference.
- **Backward Pass**: Propagates gradients only through active units.

### Network Architecture
- Modular fully connected networks supporting:
  - Arbitrary hidden layer sizes
  - L2 regularization
  - Batch normalization
  - Dropout
- Adam optimization used for training.

## Results

### Batch Normalization Benefits
- **Faster Convergence**: Networks with batch normalization achieve higher validation accuracy faster ([see training curves](./submission_logs/)).
- **Stable Training**: Enables training of deeper networks (5-6 layers) with less sensitivity to initialization.

### Dropout Regularization
- **Improved Generalization**: 0.5 dropout rate reduces overfitting gap by ~15% compared to no dropout.
- **Better Validation Performance**: Final validation accuracy improves from 29.5% (no dropout) to 32.1% with dropout.

## Usage
1. **Data Setup**: Ensure CIFAR-10 data is in `deeplearning/datasets/`
2. **Run Experiments**:
```python
# Compare BN vs non-BN networks
python train.py --use_batchnorm

# Compare different dropout rates
python train.py --dropout 0.5
```
3. **Visualize Results**:
```python
python plot_results.py --log_dir submission_logs
```

## File Structure
```
├── deeplearning/
│   ├── classifiers/       # Network implementations
│   ├── data_utils/        # CIFAR-10 loading
│   ├── gradient_check/    # Numerical gradient utilities
│   └── layers.py          # Layer implementations
├── BatchNormalization.ipynb  # Main notebook
└── submission_logs/       # Training history records
```

## Acknowledgments
This project uses starter code from **UC Berkeley's CS182/282A course** (assignment framework, data loading utilities, and solver class). The core implementations of batch normalization and dropout, along with experimental analysis, were developed independently.

## References
- Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
- Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
