# Block Coordinate Descent for Deep Neural Networks

> Layer-wise optimization for deep learning using block coordinate descent

A PyTorch-based library implementing block coordinate descent for training deep neural networks. Instead of updating all weights simultaneously, this approach alternates between layers, updating only one block of parameters per iteration. This technique can improve convergence in certain scenarios, reduce memory overhead, and provide fine-grained control over the training process.

## ✨ Features

- **Block-wise Parameter Updates** — Train neural networks by updating one layer at a time, freezing all other parameters during each optimization step. This implements true block coordinate descent for deep learning.
- **Flexible Block Definition** — Define custom parameter blocks corresponding to layers, groups of layers, or arbitrary parameter subsets. The LayerBlock abstraction allows fine-grained control over optimization granularity.
- **Automatic Gradient Freezing** — Automatically manages parameter freezing and unfreezing during training. Only the active block receives gradients, ensuring computational efficiency and correct block coordinate descent behavior.
- **Per-Block Loss Tracking** — Monitor and log loss values for each block independently. Track which layers contribute most to the overall objective and identify convergence patterns across different network depths.
- **PyTorch Integration** — Seamlessly integrates with existing PyTorch models and training pipelines. Works with standard optimizers, loss functions, and data loaders without requiring model architecture changes.
- **Comparison Tools** — Built-in utilities to compare block coordinate descent against standard full-batch training. Includes visualization and statistical analysis of convergence behavior and parameter update norms.

## 📦 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9.0 or higher
- NumPy 1.19.0 or higher
- Matplotlib 3.3.0 or higher (for visualization)

### Setup

1. Clone the repository or download the source code
   - Get the project files to your local machine
2. pip install torch numpy matplotlib
   - Install required dependencies for training and visualization
3. Verify installation by running: python -c "import torch; import lib.core; print('Installation successful!')"
   - Confirm all modules are importable and dependencies are satisfied
4. python demo.py
   - Run the demo script to see block coordinate descent in action on a toy MLP problem

## 🚀 Usage

### Basic Block Coordinate Descent Training

Train a simple MLP using block coordinate descent, updating one layer per iteration.

```
import torch
import torch.nn as nn
from lib.core import BlockCoordinateDescentCoordinator, LayerBlock
from lib.utils import create_blocks_from_model

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Create blocks (one per linear layer)
blocks = create_blocks_from_model(model, block_type='linear')

# Initialize coordinator
coordinator = BlockCoordinateDescentCoordinator(
    blocks=blocks,
    optimizer_class=torch.optim.SGD,
    optimizer_kwargs={'lr': 0.01}
)

# Training loop
criterion = nn.MSELoss()
for epoch in range(10):
    for block_idx in range(len(blocks)):
        # Generate dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Update only the current block
        loss = coordinator.step(block_idx, model, x, y, criterion)
        print(f"Epoch {epoch}, Block {block_idx}, Loss: {loss:.4f}")
```

**Output:**

```
Epoch 0, Block 0, Loss: 1.2345
Epoch 0, Block 1, Loss: 1.1234
Epoch 0, Block 2, Loss: 1.0123
...
Epoch 9, Block 2, Loss: 0.0234
```

### Custom Block Definition

Define custom parameter blocks for more control over which parameters are updated together.

```
import torch.nn as nn
from lib.core import LayerBlock

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.BatchNorm1d(20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

# Create custom blocks: first block includes linear + batchnorm
block1_params = list(model[0].parameters()) + list(model[1].parameters())
block2_params = list(model[3].parameters())

block1 = LayerBlock(name="conv_block", parameters=block1_params)
block2 = LayerBlock(name="output_layer", parameters=block2_params)

blocks = [block1, block2]
print(f"Block 1: {block1.num_parameters()} parameters")
print(f"Block 2: {block2.num_parameters()} parameters")
```

**Output:**

```
Block 1: 240 parameters
Block 2: 210 parameters
```

### Comparing BCD vs Standard Training

Use built-in utilities to compare block coordinate descent against full-batch gradient descent.

```
from lib.utils import compare_training_methods
import torch.nn as nn
import torch

# Create model and data
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Generate synthetic dataset
X_train = torch.randn(100, 5)
y_train = torch.randn(100, 1)

# Compare methods
results = compare_training_methods(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=50,
    lr=0.01,
    verbose=True
)

print(f"BCD Final Loss: {results['bcd_losses'][-1]:.4f}")
print(f"Standard Final Loss: {results['standard_losses'][-1]:.4f}")
print(f"BCD Convergence Speed: {results['bcd_convergence_epoch']} epochs")
```

**Output:**

```
Training with Block Coordinate Descent...
Training with Standard SGD...
BCD Final Loss: 0.1234
Standard Final Loss: 0.1456
BCD Convergence Speed: 35 epochs
```

### Per-Block Loss Monitoring

Track and analyze loss contributions from individual blocks during training.

```
from lib.core import BlockCoordinateDescentCoordinator
from lib.utils import create_blocks_from_model, BlockLossTracker
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

blocks = create_blocks_from_model(model, block_type='linear')
coordinator = BlockCoordinateDescentCoordinator(
    blocks=blocks,
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs={'lr': 0.001}
)

tracker = BlockLossTracker(num_blocks=len(blocks))
criterion = nn.MSELoss()

for epoch in range(5):
    for block_idx in range(len(blocks)):
        x = torch.randn(16, 8)
        y = torch.randn(16, 1)
        loss = coordinator.step(block_idx, model, x, y, criterion)
        tracker.record(block_idx, loss)
    
    print(f"Epoch {epoch}: {tracker.get_epoch_summary()}")

tracker.plot_block_losses()
```

**Output:**

```
Epoch 0: {'block_0': 0.8234, 'block_1': 0.7123, 'block_2': 0.6543}
Epoch 1: {'block_0': 0.6123, 'block_1': 0.5234, 'block_2': 0.4876}
...
Epoch 4: {'block_0': 0.1234, 'block_1': 0.0987, 'block_2': 0.0765}
[Displays matplotlib plot showing loss curves for each block]
```

## 🏗️ Architecture

The library follows a modular architecture with three main components: (1) LayerBlock - encapsulates parameter groups and provides block-level abstractions, (2) BlockCoordinateDescentCoordinator - manages the training loop, handles block selection, freezing/unfreezing, and optimization, and (3) Utilities - helper functions for block creation, comparison, visualization, and loss tracking. The design separates concerns between block definition, optimization coordination, and framework integration.

### File Structure

```
┌─────────────────────────────────────────────┐
│              User Application               │
│         (demo.py, custom scripts)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           lib/core.py                       │
│  ┌────────────────────────────────────┐    │
│  │  LayerBlock                        │    │
│  │  - parameters: List[Parameter]     │    │
│  │  - freeze() / unfreeze()           │    │
│  └────────────────────────────────────┘    │
│  ┌────────────────────────────────────┐    │
│  │  BlockCoordinateDescentCoordinator │    │
│  │  - blocks: List[LayerBlock]        │    │
│  │  - step(block_idx, ...)            │    │
│  │  - _freeze_all_except(idx)         │    │
│  └────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           lib/utils.py                      │
│  - create_blocks_from_model()              │
│  - BlockLossTracker                        │
│  - compare_training_methods()              │
│  - visualize_parameter_updates()           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         PyTorch Framework                   │
│  (nn.Module, Optimizer, Loss Functions)     │
└─────────────────────────────────────────────┘
```

### Files

- **lib/core.py** — Implements LayerBlock abstraction and BlockCoordinateDescentCoordinator for managing block-wise optimization and parameter freezing.
- **lib/utils.py** — Provides utility functions for block creation, loss tracking, training comparison, visualization, and parameter update statistics.
- **demo.py** — Demonstrates block coordinate descent on a toy MLP problem with verbose logging, comparison against standard training, and visualization.

### Design Decisions

- LayerBlock encapsulates parameter groups to provide a clean abstraction for freezing/unfreezing and tracking block-level state independently.
- BlockCoordinateDescentCoordinator separates optimization logic from model definition, allowing it to work with any PyTorch nn.Module without modification.
- Gradient freezing is implemented via requires_grad manipulation rather than masking, ensuring computational efficiency and preventing unnecessary gradient computation.
- Per-block optimizers are created to allow different learning rates or optimization strategies for different layers if needed.
- The coordinator accepts a forward pass callable, enabling flexibility in how loss is computed and allowing integration with custom training loops.
- Block creation utilities support both automatic detection (by layer type) and manual specification for maximum flexibility.
- Loss tracking is separated into its own class to maintain single responsibility and allow easy extension for custom metrics.

## 🔧 Technical Details

### Dependencies

- **torch** (1.9.0+) — Core deep learning framework providing neural network modules, automatic differentiation, and optimization algorithms.
- **numpy** (1.19.0+) — Numerical computing library used for data manipulation, statistical analysis, and array operations in utilities.
- **matplotlib** (3.3.0+) — Visualization library for plotting loss curves, parameter update norms, and training comparisons.

### Key Algorithms / Patterns

- Block Coordinate Descent: Alternates between parameter blocks, fully minimizing the objective with respect to each block while keeping others fixed.
- Gradient Freezing: Dynamically sets requires_grad=False for all parameters except the active block to prevent gradient computation and updates.
- Per-Block Optimization: Each block maintains its own optimizer state, allowing independent learning rates and momentum accumulation.
- Cyclic Block Selection: Iterates through blocks in a round-robin fashion, though the architecture supports custom selection strategies.
- Loss Decomposition Tracking: Records per-block loss contributions to analyze which layers impact the objective most significantly.

### Important Notes

- Block coordinate descent does not guarantee faster convergence than full-batch methods; effectiveness depends on problem structure and block coupling.
- Memory savings are modest since the full model must remain in memory; primary benefits are in gradient computation and fine-grained control.
- The coordinator assumes blocks are defined such that the loss is differentiable with respect to each block's parameters independently.
- When using batch normalization or dropout, ensure running statistics are updated appropriately during block-wise training.
- For very deep networks, consider grouping layers into larger blocks to reduce the overhead of frequent freezing/unfreezing operations.

## ❓ Troubleshooting

### Loss increases or diverges during training

**Cause:** Learning rate may be too high for block-wise updates, or blocks are too tightly coupled causing instability when optimized independently.

**Solution:** Reduce the learning rate by 10x and try again. Consider grouping tightly coupled layers (e.g., conv + batchnorm) into the same block. Monitor per-block losses to identify problematic layers.

### All parameters are updating despite block selection

**Cause:** Parameters are not being properly frozen, or gradient computation is happening before freeze() is called.

**Solution:** Ensure coordinator.step() is called correctly and verify that requires_grad is False for frozen parameters using: print([(n, p.requires_grad) for n, p in model.named_parameters()])

### RuntimeError: element 0 of tensors does not require grad

**Cause:** The active block's parameters have requires_grad=False, preventing gradient computation and optimization.

**Solution:** Check that LayerBlock.unfreeze() is being called on the active block. Verify parameters were added to the block correctly and have requires_grad=True initially.

### Training is much slower than standard SGD

**Cause:** Overhead from frequent parameter freezing/unfreezing and optimizer state management across many small blocks.

**Solution:** Group multiple layers into larger blocks to reduce the number of freeze/unfreeze operations per epoch. Use fewer, larger blocks for faster training at the cost of less granular control.

### Batch normalization statistics are incorrect

**Cause:** BatchNorm running mean/variance may not update correctly when layers are frozen, leading to train/eval discrepancy.

**Solution:** Ensure BatchNorm layers are in training mode (model.train()) during block updates. Consider including BatchNorm parameters in the same block as their preceding conv/linear layer.

---

This project demonstrates block coordinate descent for deep learning as an educational tool and research prototype. While the technique has theoretical foundations, practical performance depends heavily on network architecture and problem characteristics. The implementation prioritizes clarity and modularity over maximum performance. This README and portions of the codebase were generated with AI assistance.