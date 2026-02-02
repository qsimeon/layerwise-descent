"""
Block Coordinate Descent Training Demo
=======================================

This demo implements block coordinate descent for training deep neural networks,
where instead of updating all weights simultaneously, we update one layer (or block)
at a time in a cyclic or strategic manner.

This approach can be useful for:
- Memory-constrained environments
- Analyzing layer-wise training dynamics
- Exploring alternative optimization strategies
- Reducing computational overhead per iteration
"""

import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Import from our custom modules
from core import LayerSelectionStrategy, BlockCoordinateDescentTrainer
from utils import (
    filter_linear_layers,
    filter_conv_layers,
    count_parameters,
    count_layer_parameters,
    get_layer_gradients,
    get_layer_weight_norms,
    analyze_layer_updates,
    create_layer_schedule,
    combine_filters
)


# ============================================================================
# Define a simple neural network for demonstration
# ============================================================================

class SimpleDeepNet(nn.Module):
    """A simple deep neural network for demonstration purposes."""
    
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [256, 128, 64], output_dim: int = 10):
        super(SimpleDeepNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f'fc{i+1}', nn.Linear(prev_dim, hidden_dim)))
            layers.append((f'relu{i+1}', nn.ReLU()))
            prev_dim = hidden_dim
        
        layers.append((f'fc{len(hidden_dims)+1}', nn.Linear(prev_dim, output_dim)))
        
        self.network = nn.Sequential(nn.ModuleDict(dict(layers)))
    
    def forward(self, x):
        return self.network(x)


class ConvNet(nn.Module):
    """A simple convolutional neural network for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_synthetic_classification_data(
    num_samples: int = 1000,
    input_dim: int = 784,
    num_classes: int = 10,
    batch_size: int = 32
) -> tuple:
    """Generate synthetic classification data for demonstration."""
    
    # Generate random features
    X = torch.randn(num_samples, input_dim)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def generate_synthetic_image_data(
    num_samples: int = 1000,
    image_size: int = 28,
    num_classes: int = 10,
    batch_size: int = 32
) -> tuple:
    """Generate synthetic image data for CNN demonstration."""
    
    # Generate random images (1 channel, 28x28)
    X = torch.randn(num_samples, 1, image_size, image_size)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_comparison(
    standard_losses: List[float],
    bcd_losses: List[float],
    title: str = "Training Loss Comparison"
):
    """Plot comparison between standard and BCD training."""
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(standard_losses, label='Standard SGD', linewidth=2)
    plt.plot(bcd_losses, label='Block Coordinate Descent', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss difference
    plt.subplot(1, 2, 2)
    if len(standard_losses) == len(bcd_losses):
        diff = np.array(standard_losses) - np.array(bcd_losses)
        plt.plot(diff, linewidth=2, color='green')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference (Standard - BCD)')
        plt.title('Training Difference')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved training comparison plot to 'training_comparison.png'")
    plt.close()


def plot_layer_update_analysis(update_history: List[str], layer_names: List[str]):
    """Visualize which layers were updated over time."""
    
    # Analyze update history
    analysis = analyze_layer_updates(update_history)
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Update frequency per layer
    plt.subplot(1, 2, 1)
    update_counts = analysis['update_counts']
    layers = list(update_counts.keys())
    counts = list(update_counts.values())
    
    plt.barh(layers, counts, color='steelblue')
    plt.xlabel('Number of Updates')
    plt.ylabel('Layer Name')
    plt.title('Layer Update Frequency')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Update timeline
    plt.subplot(1, 2, 2)
    layer_to_idx = {name: idx for idx, name in enumerate(layer_names)}
    update_indices = [layer_to_idx.get(layer, -1) for layer in update_history[:100]]
    
    plt.scatter(range(len(update_indices)), update_indices, alpha=0.6, s=20)
    plt.xlabel('Iteration')
    plt.ylabel('Layer Index')
    plt.title('Layer Update Timeline (First 100 iterations)')
    plt.yticks(range(len(layer_names)), layer_names)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('layer_update_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved layer update analysis to 'layer_update_analysis.png'")
    plt.close()


# ============================================================================
# Training Functions
# ============================================================================

def train_standard_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> tuple:
    """Train model using standard SGD (all weights updated simultaneously)."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_accuracies = []
    
    print("\n" + "="*60)
    print("Training with STANDARD SGD (all weights updated)")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    return train_losses, test_accuracies


# ============================================================================
# Demo Scenarios
# ============================================================================

def demo_cyclic_bcd():
    """
    Demo 1: Cyclic Block Coordinate Descent
    Updates layers in a fixed cyclic order.
    """
    print("\n" + "="*70)
    print("DEMO 1: CYCLIC BLOCK COORDINATE DESCENT")
    print("="*70)
    print("Strategy: Update layers in a fixed cyclic order (layer1 -> layer2 -> ...)")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleDeepNet(input_dim=784, hidden_dims=[256, 128, 64], output_dim=10)
    print(f"\nModel architecture: {count_parameters(model)} total parameters")
    
    # Show layer parameters
    layer_params = count_layer_parameters(model)
    print("\nLayer parameters:")
    for name, count in layer_params.items():
        print(f"  {name}: {count} parameters")
    
    # Generate data
    train_loader, test_loader = generate_synthetic_classification_data(
        num_samples=1000, input_dim=784, num_classes=10, batch_size=32
    )
    
    # Train with standard SGD for comparison
    model_standard = SimpleDeepNet(input_dim=784, hidden_dims=[256, 128, 64], output_dim=10)
    standard_losses, standard_accs = train_standard_sgd(
        model_standard, train_loader, test_loader, num_epochs=10, device=device
    )
    
    # Train with BCD
    print("\n" + "="*60)
    print("Training with BLOCK COORDINATE DESCENT (Cyclic)")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Only update linear layers
    layer_filter = filter_linear_layers
    
    trainer = BlockCoordinateDescentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        layer_filter=layer_filter,
        strategy=LayerSelectionStrategy.CYCLIC,
        device=device
    )
    
    bcd_losses = []
    bcd_accs = []
    
    for epoch in range(10):
        avg_loss, update_history = trainer.train_epoch(train_loader, epoch)
        bcd_losses.append(avg_loss)
        
        accuracy = trainer.evaluate(test_loader)
        bcd_accs.append(accuracy)
        
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    # Analyze and visualize
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    trainable_layers = trainer.get_trainable_layers()
    print(f"\nTrainable layers: {len(trainable_layers)}")
    for layer_name in trainable_layers:
        print(f"  - {layer_name}")
    
    # Plot comparison
    plot_training_comparison(standard_losses, bcd_losses, "Cyclic BCD vs Standard SGD")
    
    # Plot layer updates
    plot_layer_update_analysis(update_history, trainable_layers)
    
    print("\n✅ Demo 1 completed successfully!")
    return trainer, update_history


def demo_random_bcd():
    """
    Demo 2: Random Block Coordinate Descent
    Randomly selects which layer to update at each iteration.
    """
    print("\n" + "="*70)
    print("DEMO 2: RANDOM BLOCK COORDINATE DESCENT")
    print("="*70)
    print("Strategy: Randomly select a layer to update at each iteration")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SimpleDeepNet(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)
    
    # Generate data
    train_loader, test_loader = generate_synthetic_classification_data(
        num_samples=1000, input_dim=784, num_classes=10, batch_size=32
    )
    
    # Setup trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainer = BlockCoordinateDescentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        layer_filter=filter_linear_layers,
        strategy=LayerSelectionStrategy.RANDOM,
        device=device
    )
    
    print("\nTraining with random layer selection...")
    
    losses = []
    for epoch in range(10):
        avg_loss, update_history = trainer.train_epoch(train_loader, epoch)
        losses.append(avg_loss)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    # Analyze randomness
    analysis = analyze_layer_updates(update_history)
    print("\n" + "="*60)
    print("RANDOMNESS ANALYSIS")
    print("="*60)
    print(f"Total updates: {analysis['total_updates']}")
    print(f"Unique layers updated: {analysis['unique_layers']}")
    print("\nUpdate distribution:")
    for layer, count in analysis['update_counts'].items():
        percentage = (count / analysis['total_updates']) * 100
        print(f"  {layer}: {count} updates ({percentage:.1f}%)")
    
    print("\n✅ Demo 2 completed successfully!")
    return trainer


def demo_gradient_based_bcd():
    """
    Demo 3: Gradient-based Block Coordinate Descent
    Selects the layer with largest gradient magnitude to update.
    """
    print("\n" + "="*70)
    print("DEMO 3: GRADIENT-BASED BLOCK COORDINATE DESCENT")
    print("="*70)
    print("Strategy: Update the layer with the largest gradient magnitude")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SimpleDeepNet(input_dim=784, hidden_dims=[256, 128, 64], output_dim=10)
    
    # Generate data
    train_loader, test_loader = generate_synthetic_classification_data(
        num_samples=1000, input_dim=784, num_classes=10, batch_size=32
    )
    
    # Setup trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainer = BlockCoordinateDescentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        layer_filter=filter_linear_layers,
        strategy=LayerSelectionStrategy.GRADIENT_BASED,
        device=device
    )
    
    print("\nTraining with gradient-based layer selection...")
    
    losses = []
    for epoch in range(10):
        avg_loss, update_history = trainer.train_epoch(train_loader, epoch)
        losses.append(avg_loss)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    # Analyze which layers were prioritized
    analysis = analyze_layer_updates(update_history)
    print("\n" + "="*60)
    print("GRADIENT-BASED SELECTION ANALYSIS")
    print("="*60)
    print("Layers ranked by update frequency (most updated first):")
    sorted_layers = sorted(
        analysis['update_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for layer, count in sorted_layers:
        percentage = (count / analysis['total_updates']) * 100
        print(f"  {layer}: {count} updates ({percentage:.1f}%)")
    
    print("\n✅ Demo 3 completed successfully!")
    return trainer


def demo_scheduled_bcd():
    """
    Demo 4: Scheduled Block Coordinate Descent
    Uses a predetermined schedule for layer updates.
    """
    print("\n" + "="*70)
    print("DEMO 4: SCHEDULED BLOCK COORDINATE DESCENT")
    print("="*70)
    print("Strategy: Follow a predetermined schedule for layer updates")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SimpleDeepNet(input_dim=784, hidden_dims=[256, 128, 64], output_dim=10)
    
    # Get layer names
    trainable_layers = [
        name for name, module in model.named_modules()
        if filter_linear_layers(name, module)
    ]
    
    # Create a custom schedule: focus more on early layers initially
    print("\nCreating custom schedule...")
    schedule = create_layer_schedule(
        trainable_layers,
        schedule_type='weighted',
        num_iterations=100,
        weights=[0.4, 0.3, 0.2, 0.1]  # More updates for earlier layers
    )
    
    print(f"Schedule created with {len(schedule)} iterations")
    
    # Generate data
    train_loader, test_loader = generate_synthetic_classification_data(
        num_samples=1000, input_dim=784, num_classes=10, batch_size=32
    )
    
    # Setup trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainer = BlockCoordinateDescentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        layer_filter=filter_linear_layers,
        strategy=LayerSelectionStrategy.SCHEDULED,
        device=device,
        schedule=schedule
    )
    
    print("\nTraining with scheduled layer updates...")
    
    losses = []
    for epoch in range(10):
        avg_loss, update_history = trainer.train_epoch(train_loader, epoch)
        losses.append(avg_loss)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    # Analyze schedule adherence
    analysis = analyze_layer_updates(update_history)
    print("\n" + "="*60)
    print("SCHEDULE ANALYSIS")
    print("="*60)
    print("Actual update distribution vs intended schedule:")
    for layer, count in analysis['update_counts'].items():
        percentage = (count / analysis['total_updates']) * 100
        print(f"  {layer}: {count} updates ({percentage:.1f}%)")
    
    print("\n✅ Demo 4 completed successfully!")
    return trainer


def demo_conv_net_bcd():
    """
    Demo 5: BCD with Convolutional Neural Network
    Demonstrates BCD on a CNN, updating only convolutional layers.
    """
    print("\n" + "="*70)
    print("DEMO 5: BLOCK COORDINATE DESCENT WITH CNN")
    print("="*70)
    print("Strategy: Apply BCD to convolutional layers only")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create CNN model
    model = ConvNet(num_classes=10)
    print(f"\nCNN Model: {count_parameters(model)} total parameters")
    
    # Show layer breakdown
    layer_params = count_layer_parameters(model)
    print("\nLayer parameters:")
    for name, count in layer_params.items():
        print(f"  {name}: {count} parameters")
    
    # Generate image data
    train_loader, test_loader = generate_synthetic_image_data(
        num_samples=1000, image_size=28, num_classes=10, batch_size=32
    )
    
    # Setup trainer - only update convolutional layers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainer = BlockCoordinateDescentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        layer_filter=filter_conv_layers,  # Only conv layers
        strategy=LayerSelectionStrategy.CYCLIC,
        device=device
    )
    
    trainable_layers = trainer.get_trainable_layers()
    print(f"\nTrainable layers (conv only): {trainable_layers}")
    
    print("\nTraining CNN with BCD on convolutional layers...")
    
    losses = []
    for epoch in range(10):
        avg_loss, update_history = trainer.train_epoch(train_loader, epoch)
        losses.append(avg_loss)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    print("\n✅ Demo 5 completed successfully!")
    return trainer


def demo_combined_filters():
    """
    Demo 6: BCD with Combined Layer Filters
    Demonstrates using multiple filters to select specific layer types.
    """
    print("\n" + "="*70)
    print("DEMO 6: BLOCK COORDINATE DESCENT WITH COMBINED FILTERS")
    print("="*70)
    print("Strategy: Update both convolutional AND linear layers")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create CNN model
    model = ConvNet(num_classes=10)
    
    # Generate data
    train_loader, test_loader = generate_synthetic_image_data(
        num_samples=1000, image_size=28, num_classes=10, batch_size=32
    )
    
    # Combine filters to update both conv and linear layers
    combined_filter = combine_filters(
        filter_conv_layers,
        filter_linear_layers,
        mode='or'
    )
    
    # Setup trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainer = BlockCoordinateDescentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        layer_filter=combined_filter,
        strategy=LayerSelectionStrategy.RANDOM,
        device=device
    )
    
    trainable_layers = trainer.get_trainable_layers()
    print(f"\nTrainable layers (conv + linear): {trainable_layers}")
    
    print("\nTraining with combined filter...")
    
    losses = []
    for epoch in range(10):
        avg_loss, update_history = trainer.train_epoch(train_loader, epoch)
        losses.append(avg_loss)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    # Analyze which layer types were updated
    analysis = analyze_layer_updates(update_history)
    print("\n" + "="*60)
    print("LAYER TYPE ANALYSIS")
    print("="*60)
    
    conv_updates = sum(count for layer, count in analysis['update_counts'].items() if 'conv' in layer)
    fc_updates = sum(count for layer, count in analysis['update_counts'].items() if 'fc' in layer)
    total = analysis['total_updates']
    
    print(f"Convolutional layer updates: {conv_updates} ({100*conv_updates/total:.1f}%)")
    print(f"Linear layer updates: {fc_updates} ({100*fc_updates/total:.1f}%)")
    
    print("\n✅ Demo 6 completed successfully!")
    return trainer


# ============================================================================
# Main Demo Runner
# ============================================================================

def main():
    """Run all demonstration scenarios."""
    
    print("="*70)
    print("BLOCK COORDINATE DESCENT FOR DEEP NEURAL NETWORKS")
    print("="*70)
    print("\nThis demo showcases different strategies for training neural networks")
    print("using block coordinate descent, where only one layer is updated per")
    print("iteration instead of updating all weights simultaneously.")
    print("\nBenefits:")
    print("  • Reduced memory footprint per iteration")
    print("  • Better understanding of layer-wise training dynamics")
    print("  • Potential for parallelization across layers")
    print("  • Alternative optimization strategy for difficult problems")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run all demos
        print("\n" + "="*70)
        print("Running 6 demonstration scenarios...")
        print("="*70)
        
        # Demo 1: Cyclic BCD
        trainer1, history1 = demo_cyclic_bcd()
        
        # Demo 2: Random BCD
        trainer2 = demo_random_bcd()
        
        # Demo 3: Gradient-based BCD
        trainer3 = demo_gradient_based_bcd()
        
        # Demo 4: Scheduled BCD
        trainer4 = demo_scheduled_bcd()
        
        # Demo 5: CNN with BCD
        trainer5 = demo_conv_net_bcd()
        
        # Demo 6: Combined filters
        trainer6 = demo_combined_filters()
        
        # Final summary
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print("\nSummary:")
        print("  ✅ Demo 1: Cyclic BCD - Updates layers in fixed order")
        print("  ✅ Demo 2: Random BCD - Randomly selects layers")
        print("  ✅ Demo 3: Gradient-based BCD - Updates layer with largest gradient")
        print("  ✅ Demo 4: Scheduled BCD - Follows predetermined schedule")
        print("  ✅ Demo 5: CNN BCD - Applies BCD to convolutional networks")
        print("  ✅ Demo 6: Combined Filters - Updates multiple layer types")
        
        print("\n📊 Generated visualizations:")
        print("  • training_comparison.png - Loss comparison plots")
        print("  • layer_update_analysis.png - Layer update patterns")
        
        print("\n💡 Key Takeaways:")
        print("  • BCD can be competitive with standard SGD")
        print("  • Different strategies suit different network architectures")
        print("  • Layer selection strategy significantly impacts convergence")
        print("  • Useful for memory-constrained environments")
        
    except Exception as e:
        print(f"\n❌ Error during demo execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
