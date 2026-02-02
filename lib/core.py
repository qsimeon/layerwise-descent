"""Core module for block coordinate descent training of deep neural networks.

This module provides the main BlockCoordinateDescentTrainer class that implements
block coordinate descent optimization for neural networks, where only one layer's
weights are updated per training iteration.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable, Dict, Any, Tuple
import random
from enum import Enum


class LayerSelectionStrategy(Enum):
    """Enumeration of layer selection strategies for block coordinate descent."""
    CYCLIC = "cyclic"
    RANDOM = "random"
    GREEDY = "greedy"
    WEIGHTED_RANDOM = "weighted_random"


class BlockCoordinateDescentTrainer:
    """Trainer that implements block coordinate descent for deep neural networks.
    
    In block coordinate descent, only one block (layer) of parameters is updated
    per iteration while keeping all other parameters fixed. This can lead to
    more stable training in some scenarios and reduced memory usage.
    
    Attributes:
        model: The neural network model to train.
        criterion: Loss function.
        layer_optimizers: Dictionary mapping layer names to their optimizers.
        selection_strategy: Strategy for selecting which layer to update.
        trainable_layers: List of layer names that can be updated.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        selection_strategy: LayerSelectionStrategy = LayerSelectionStrategy.CYCLIC,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ):
        """Initialize the BlockCoordinateDescentTrainer.
        
        Args:
            model: PyTorch neural network model.
            criterion: Loss function (e.g., nn.CrossEntropyLoss()).
            optimizer_class: Optimizer class to use for each layer.
            optimizer_kwargs: Keyword arguments to pass to optimizer constructor.
            selection_strategy: Strategy for selecting layers to update.
            layer_filter: Optional function to filter which layers are trainable.
                         Should return True for layers to include.
        """
        self.model = model
        self.criterion = criterion
        self.selection_strategy = selection_strategy
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.001}
        
        # Identify trainable layers and create separate optimizers
        self.trainable_layers: List[str] = []
        self.layer_optimizers: Dict[str, torch.optim.Optimizer] = {}
        
        for name, module in model.named_modules():
            # Skip empty names and modules without parameters
            if not name or not list(module.parameters(recurse=False)):
                continue
            
            # Apply filter if provided
            if layer_filter and not layer_filter(name, module):
                continue
            
            # Get parameters for this specific layer (non-recursive)
            layer_params = list(module.parameters(recurse=False))
            if layer_params:
                self.trainable_layers.append(name)
                self.layer_optimizers[name] = optimizer_class(
                    layer_params,
                    **self.optimizer_kwargs
                )
        
        # State for cyclic selection
        self._current_layer_idx = 0
        
        # State for weighted random selection
        self._layer_weights: Dict[str, float] = {
            name: 1.0 for name in self.trainable_layers
        }
        
        # Freeze all parameters initially
        self._freeze_all_layers()
    
    def _freeze_all_layers(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _freeze_layer(self, layer_name: str) -> None:
        """Freeze parameters of a specific layer.
        
        Args:
            layer_name: Name of the layer to freeze.
        """
        for name, module in self.model.named_modules():
            if name == layer_name:
                for param in module.parameters(recurse=False):
                    param.requires_grad = False
                break
    
    def _unfreeze_layer(self, layer_name: str) -> None:
        """Unfreeze parameters of a specific layer.
        
        Args:
            layer_name: Name of the layer to unfreeze.
        """
        for name, module in self.model.named_modules():
            if name == layer_name:
                for param in module.parameters(recurse=False):
                    param.requires_grad = True
                break
    
    def _select_layer_cyclic(self) -> str:
        """Select layer using cyclic strategy.
        
        Returns:
            Name of the selected layer.
        """
        layer_name = self.trainable_layers[self._current_layer_idx]
        self._current_layer_idx = (self._current_layer_idx + 1) % len(self.trainable_layers)
        return layer_name
    
    def _select_layer_random(self) -> str:
        """Select layer using random strategy.
        
        Returns:
            Name of the selected layer.
        """
        return random.choice(self.trainable_layers)
    
    def _select_layer_weighted_random(self) -> str:
        """Select layer using weighted random strategy.
        
        Returns:
            Name of the selected layer.
        """
        layers = list(self._layer_weights.keys())
        weights = [self._layer_weights[layer] for layer in layers]
        return random.choices(layers, weights=weights, k=1)[0]
    
    def select_layer(self) -> str:
        """Select the next layer to update based on the selection strategy.
        
        Returns:
            Name of the selected layer.
        """
        if self.selection_strategy == LayerSelectionStrategy.CYCLIC:
            return self._select_layer_cyclic()
        elif self.selection_strategy == LayerSelectionStrategy.RANDOM:
            return self._select_layer_random()
        elif self.selection_strategy == LayerSelectionStrategy.WEIGHTED_RANDOM:
            return self._select_layer_weighted_random()
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def set_layer_weight(self, layer_name: str, weight: float) -> None:
        """Set the selection weight for a layer (used in weighted random strategy).
        
        Args:
            layer_name: Name of the layer.
            weight: Selection weight (higher = more likely to be selected).
        """
        if layer_name in self._layer_weights:
            self._layer_weights[layer_name] = weight
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> Tuple[float, str]:
        """Perform a single training step, updating only one layer.
        
        Args:
            inputs: Input batch tensor.
            targets: Target batch tensor.
            layer_name: Specific layer to update. If None, selects automatically.
        
        Returns:
            Tuple of (loss value, name of updated layer).
        """
        # Select layer to update
        if layer_name is None:
            layer_name = self.select_layer()
        
        # Ensure all layers are frozen
        self._freeze_all_layers()
        
        # Unfreeze selected layer
        self._unfreeze_layer(layer_name)
        
        # Get optimizer for this layer
        optimizer = self.layer_optimizers[layer_name]
        
        # Forward pass
        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        return loss.item(), layer_name
    
    def train_epoch(
        self,
        dataloader: Any,
        device: str = "cpu",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Train for one complete epoch.
        
        Args:
            dataloader: DataLoader providing batches of (inputs, targets).
            device: Device to use for training ('cpu' or 'cuda').
            verbose: Whether to print progress information.
        
        Returns:
            Dictionary containing training statistics.
        """
        self.model.to(device)
        total_loss = 0.0
        num_batches = 0
        layer_update_counts = {name: 0 for name in self.trainable_layers}
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            loss, updated_layer = self.train_step(inputs, targets)
            total_loss += loss
            num_batches += 1
            layer_update_counts[updated_layer] += 1
            
            if verbose and batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss:.4f}, Updated layer = {updated_layer}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "avg_loss": avg_loss,
            "total_batches": num_batches,
            "layer_update_counts": layer_update_counts
        }
    
    def evaluate(
        self,
        dataloader: Any,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader providing batches of (inputs, targets).
            device: Device to use for evaluation.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.to(device)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy for classification tasks
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        return {
            "avg_loss": avg_loss,
            "accuracy": accuracy
        }
    
    def get_trainable_layers(self) -> List[str]:
        """Get list of trainable layer names.
        
        Returns:
            List of layer names that can be updated.
        """
        return self.trainable_layers.copy()
