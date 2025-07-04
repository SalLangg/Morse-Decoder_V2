from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Tuple, Optional, Union, Dict, List
from torch.utils.tensorboard import SummaryWriter


class BaseMLModel(ABC):
    """Abstract base class for models"""
    @property
    def device(self) -> torch.device:
        """Get the devicethe model"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def name(self) -> str:
        """Get the model name"""
        pass

    @abstractmethod
    def fit(self, data: np.ndarray, targets: np.ndarray) -> Dict[str, list]:
        """Train the model on data with targets
        Returns:
            Dictionary of training and validating loss
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on data"""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk"""
        pass


    @staticmethod
    def remove(path: Union[str, Path]) -> bool:
        """Delete model file from disk
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f'Error deleting model: {e}')
            return False


