from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, Optional
import json


class Storage(ABC):
    """Abstract base class for storage implementations."""

    def __init__(self) -> None:
        """Initialize the storage."""
        pass

    @abstractmethod
    def save(self, data: Any, path: Union[str, Path], **kwargs) -> bool:
        """
        Save data to the specified path.
        
        Args:
            data: The data to save
            path: The path where to save the data
            **kwargs: Additional arguments specific to the storage implementation
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """
        Load data from the specified path.
        
        Args:
            path: The path from where to load the data
            **kwargs: Additional arguments specific to the storage implementation
            
        Returns:
            Any: The loaded data
        """
        pass

    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if a file exists at the specified path.
        
        Args:
            path: The path to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, path: Union[str, Path]) -> bool:
        """
        Delete file at the specified path.
        
        Args:
            path: The path to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def list_files(self, path: Union[str, Path], pattern: Optional[str] = None) -> list:
        """
        List files in the specified directory.
        
        Args:
            path: The directory path
            pattern: Optional glob pattern to filter files
            
        Returns:
            list: List of file paths
        """
        pass
