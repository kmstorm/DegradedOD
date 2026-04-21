"""
Feature Hook Manager for extracting intermediate representations from frozen models.

Provides a clean API to register, capture, and remove forward hooks on arbitrary
PyTorch modules without modifying their source code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HookRecord:
    """Metadata for a single registered hook."""

    name: str
    module_path: str
    handle: torch.utils.hooks.RemovableHook
    transform_fn: Callable | None = None


class FeatureHookManager:
    """
    Manages forward hooks on a PyTorch model to capture intermediate activations.

    Usage:
        manager = FeatureHookManager()
        manager.register(model, "backbone.layers.3", name="backbone_out")
        manager.register(model, "feature_enhancer", name="enhanced_features")

        output = model(**inputs)
        features = manager.get_features()
        # features == {"backbone_out": Tensor, "enhanced_features": Tensor}

        manager.clear()   # reset captured features
        manager.remove()  # remove all hooks
    """

    def __init__(self) -> None:
        self._hooks: dict[str, HookRecord] = {}
        self._features: dict[str, Any] = {}

    @property
    def feature_names(self) -> list[str]:
        """Names of all registered hooks."""
        return list(self._hooks.keys())

    def register(
        self,
        model: nn.Module,
        module_path: str,
        name: str | None = None,
        transform_fn: Callable | None = None,
    ) -> str:
        """
        Register a forward hook on a specific submodule.

        Args:
            model: The parent model.
            module_path: Dot-separated path to the target submodule
                         (e.g., "backbone.layers.3" or "model.feature_enhancer").
            name: Optional human-readable name for this hook. Defaults to module_path.
            transform_fn: Optional callable applied to the raw output before storing.
                          Signature: transform_fn(output) -> processed_output.

        Returns:
            The name assigned to this hook.
        """
        name = name or module_path

        if name in self._hooks:
            logger.warning(f"Hook '{name}' already exists, replacing it.")
            self._hooks[name].handle.remove()

        # Navigate to the target submodule
        target_module = self._resolve_module(model, module_path)

        def _hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
            if transform_fn is not None:
                self._features[name] = transform_fn(output)
            else:
                self._features[name] = output

        handle = target_module.register_forward_hook(_hook_fn)

        self._hooks[name] = HookRecord(
            name=name,
            module_path=module_path,
            handle=handle,
            transform_fn=transform_fn,
        )

        logger.info(f"Registered hook '{name}' on module '{module_path}'")
        return name

    def get_features(self) -> dict[str, Any]:
        """
        Return all captured features from the last forward pass.

        Returns:
            Dict mapping hook names to their captured outputs.
        """
        return dict(self._features)

    def get(self, name: str) -> Any:
        """
        Get a specific captured feature by name.

        Args:
            name: The hook name.

        Returns:
            The captured output tensor/tuple.

        Raises:
            KeyError: If the hook name doesn't exist or hasn't been triggered yet.
        """
        if name not in self._features:
            available = list(self._features.keys())
            raise KeyError(
                f"Feature '{name}' not found. Available: {available}. "
                f"Ensure a forward pass has been executed after registering the hook."
            )
        return self._features[name]

    def clear(self) -> None:
        """Clear all captured features (keeps hooks registered)."""
        self._features.clear()

    def remove(self, name: str | None = None) -> None:
        """
        Remove hook(s).

        Args:
            name: If provided, remove only this hook. If None, remove all hooks.
        """
        if name is not None:
            if name in self._hooks:
                self._hooks[name].handle.remove()
                del self._hooks[name]
                self._features.pop(name, None)
                logger.info(f"Removed hook '{name}'")
            else:
                logger.warning(f"Hook '{name}' not found, nothing to remove.")
        else:
            for record in self._hooks.values():
                record.handle.remove()
            self._hooks.clear()
            self._features.clear()
            logger.info("Removed all hooks")

    def __len__(self) -> int:
        return len(self._hooks)

    def __repr__(self) -> str:
        hook_info = ", ".join(
            f"{r.name}@{r.module_path}" for r in self._hooks.values()
        )
        return f"FeatureHookManager(hooks=[{hook_info}])"

    @staticmethod
    def _resolve_module(model: nn.Module, module_path: str) -> nn.Module:
        """
        Navigate a dot-separated path to find the target submodule.

        Args:
            model: Root module.
            module_path: Dot-separated path (e.g., "backbone.layers.3").

        Returns:
            The target nn.Module.

        Raises:
            AttributeError: If the path is invalid.
        """
        parts = module_path.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
                current = current[int(part)]
            else:
                # Try named_modules as fallback
                found = False
                for child_name, child_module in current.named_children():
                    if child_name == part:
                        current = child_module
                        found = True
                        break
                if not found:
                    raise AttributeError(
                        f"Module '{module_path}' not found. "
                        f"Failed at part '{part}'. "
                        f"Available children: {[n for n, _ in current.named_children()]}"
                    )
        return current

    @staticmethod
    def list_modules(model: nn.Module, max_depth: int = 3) -> list[str]:
        """
        Utility: list all submodule paths up to a given depth.

        Useful for discovering which modules are available for hooking.

        Args:
            model: The model to inspect.
            max_depth: Maximum nesting depth to list.

        Returns:
            List of dot-separated module paths.
        """
        paths = []
        for name, _ in model.named_modules():
            if name and name.count(".") < max_depth:
                paths.append(name)
        return paths
