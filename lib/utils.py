"""Utility functions for block coordinate descent training.

This module provides helper functions for layer filtering, visualization,
and analysis of block coordinate descent training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict


def filter_linear_layers(name: str, module: nn.Module) -> bool:
    """Filter function to select only linear/fully-connected layers.
    
    Args:
        name: Name of the layer.
        module: The layer module.
    
    Returns:
        True if the layer is a Linear layer, False otherwise.
    """
    return isinstance(module, nn.Linear)


def filter_conv_layers(name: str, module: nn.Module) -> bool:
    """Filter function to select only convolutional layers.
    
    Args:
        name: Name of the layer.
        module: The layer module.
    
    Returns:
        True if the layer is a Conv layer, False otherwise.
    """
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))


def filter_normalization_layers(name: str, module: nn.Module) -> bool:
    """Filter function to select only normalization layers.
    
    Args:
        name: Name of the layer.
        module: The layer module.
    
    Returns:
        True if the layer is a normalization layer, False otherwise.
    """
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                               nn.InstanceNorm2d, nn.InstanceNorm3d))


def filter_by_name_pattern(pattern: str) -> Callable[[str, nn.Module], bool]:
    """Create a filter function that matches layer names by pattern.
    
    Args:
        pattern: String pattern to match in layer names.
    
    Returns:
        Filter function that returns True if pattern is in the layer name.
    """
    def filter_fn(name: str, module: nn.Module) -> bool:
        return pattern in name
    return filter_fn


def combine_filters(
    *filters: Callable[[str, nn.Module], bool],
    mode: str = "or"
) -> Callable[[str, nn.Module], bool]:
    """Combine multiple filter functions.
    
    Args:
        *filters: Variable number of filter functions.
        mode: Combination mode - 'or' (any filter matches) or 'and' (all filters match).
    
    Returns:
        Combined filter function.
    """
    def combined_filter(name: str, module: nn.Module) -> bool:
        results = [f(name, module) for f in filters]
        if mode == "or":
            return any(results)
        elif mode == "and":
            return all(results)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'or' or 'and'.")
    return combined_filter


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
    
    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_layer_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in each layer of the model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary mapping layer names to parameter counts.
    """
    layer_params = {}
    for name, module in model.named_modules():
        if name:  # Skip empty names
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                layer_params[name] = params
    return layer_params


def get_layer_gradients(model: nn.Module) -> Dict[str, float]:
    """Get the average gradient magnitude for each layer.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary mapping layer names to average gradient magnitudes.
    """
    layer_grads = {}
    for name, module in model.named_modules():
        if name:
            grads = []
            for param in module.parameters(recurse=False):
                if param.grad is not None:
                    grads.append(param.grad.abs().mean().item())
            if grads:
                layer_grads[name] = np.mean(grads)
    return layer_grads


def get_layer_weight_norms(model: nn.Module) -> Dict[str, float]:
    """Get the L2 norm of weights for each layer.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary mapping layer names to weight norms.
    """
    layer_norms = {}
    for name, module in model.named_modules():
        if name:
            norms = []
            for param in module.parameters(recurse=False):
                norms.append(param.data.norm(2).item())
            if norms:
                layer_norms[name] = np.sqrt(sum(n**2 for n in norms))
    return layer_norms


def analyze_layer_updates(
    update_history: List[str]
) -> Dict[str, Any]:
    """Analyze the history of layer updates.
    
    Args:
        update_history: List of layer names in the order they were updated.
    
    Returns:
        Dictionary containing analysis statistics.
    """
    if not update_history:
        return {}
    
    # Count updates per layer
    update_counts = defaultdict(int)
    for layer in update_history:
        update_counts[layer] += 1
    
    # Calculate update frequencies
    total_updates = len(update_history)
    update_frequencies = {
        layer: count / total_updates
        for layer, count in update_counts.items()
    }
    
    # Find consecutive update patterns
    consecutive_updates = defaultdict(int)
    for i in range(len(update_history) - 1):
        if update_history[i] == update_history[i + 1]:
            consecutive_updates[update_history[i]] += 1
    
    return {
        "update_counts": dict(update_counts),
        "update_frequencies": update_frequencies,
        "consecutive_updates": dict(consecutive_updates),
        "total_updates": total_updates,
        "unique_layers": len(update_counts)
    }


def create_layer_schedule(
    layer_names: List[str],
    schedule_type: str = "cyclic",
    num_iterations: int = 100,
    **kwargs
) -> List[str]:
    """Create a predetermined schedule for layer updates.
    
    Args:
        layer_names: List of available layer names.
        schedule_type: Type of schedule ('cyclic', 'random', 'reverse_cyclic').
        num_iterations: Number of iterations to schedule.
        **kwargs: Additional arguments for specific schedule types.
    
    Returns:
        List of layer names in the order they should be updated.
    """
    schedule = []
    
    if schedule_type == "cyclic":
        for i in range(num_iterations):
            schedule.append(layer_names[i % len(layer_names)])
    
    elif schedule_type == "reverse_cyclic":
        reversed_layers = list(reversed(layer_names))
        for i in range(num_iterations):
            schedule.append(reversed_layers[i % len(reversed_layers)])
    
    elif schedule_type == "random":
        import random
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)
        schedule = [random.choice(layer_names) for _ in range(num_iterations)]
    
    elif schedule_type == "block":
        # Update each layer multiple times before moving to next
        block_size = kwargs.get("block_size", 10)
        for i in range(num_iterations):
            layer_idx = (i // block_size) % len(layer_names)
            schedule.append(layer_names[layer_idx])
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return schedule


def compute_layer_importance(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: str = "cpu",
    method: str = "gradient"
) -> Dict[str, float]:
    """Compute importance scores for each layer.
    
    Args:
        model: PyTorch model.
        dataloader: DataLoader for computing importance.
        criterion: Loss function.
        device: Device to use.
        method: Method for computing importance ('gradient' or 'weight_norm').
    
    Returns:
        Dictionary mapping layer names to importance scores.
    """
    model.to(device)
    model.train()
    
    if method == "gradient":
        # Compute average gradient magnitude
        layer_grads = defaultdict(list)
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, module in model.named_modules():
                if name:
                    for param in module.parameters(recurse=False):
                        if param.grad is not None:
                            layer_grads[name].append(param.grad.abs().mean().item())
        
        # Average across batches
        importance = {
            name: np.mean(grads) if grads else 0.0
            for name, grads in layer_grads.items()
        }
    
    elif method == "weight_norm":
        importance = get_layer_weight_norms(model)
    
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    return importance


def save_training_state(
    trainer: Any,
    filepath: str,
    epoch: int,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """Save the training state including model and optimizer states.
    
    Args:
        trainer: BlockCoordinateDescentTrainer instance.
        filepath: Path to save the checkpoint.
        epoch: Current epoch number.
        additional_info: Additional information to save.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_states": {
            name: opt.state_dict()
            for name, opt in trainer.layer_optimizers.items()
        },
        "trainable_layers": trainer.trainable_layers,
        "current_layer_idx": trainer._current_layer_idx,
        "layer_weights": trainer._layer_weights,
    }
    
    if additional_info:
        state["additional_info"] = additional_info
    
    torch.save(state, filepath)


def load_training_state(
    trainer: Any,
    filepath: str
) -> Dict[str, Any]:
    """Load the training state from a checkpoint.
    
    Args:
        trainer: BlockCoordinateDescentTrainer instance.
        filepath: Path to the checkpoint file.
    
    Returns:
        Dictionary containing loaded state information.
    """
    checkpoint = torch.load(filepath)
    
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    
    for name, opt_state in checkpoint["optimizer_states"].items():
        if name in trainer.layer_optimizers:
            trainer.layer_optimizers[name].load_state_dict(opt_state)
    
    trainer._current_layer_idx = checkpoint.get("current_layer_idx", 0)
    trainer._layer_weights = checkpoint.get("layer_weights", trainer._layer_weights)
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "additional_info": checkpoint.get("additional_info", {})
    }
