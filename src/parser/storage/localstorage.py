from .storage import Storage
from pathlib import Path
from typing import Any, Union, Optional, List
import json
import shutil
import pickle
from core.logger import logger


class LocalStorage(Storage):
    """Local file system storage implementation."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize LocalStorage with optional base path.
        
        Args:
            base_path: Base directory for all storage operations
        """
        super().__init__()
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = logger.get_logger(__name__)
        self._ensure_base_path()

    def _ensure_base_path(self) -> None:
        """Ensure the base path directory exists."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Base path ensured: {self.base_path}")
        except Exception as e:
            self.logger.error(f"Failed to create base path {self.base_path}: {e}")
            raise

    def _get_full_path(self, path: Union[str, Path]) -> Path:
        """Get the full path by combining with base path if relative."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path

    def save(self, data: Any, path: Union[str, Path], **kwargs) -> bool:
        """
        Save data to local file system.
        
        Args:
            data: Data to save (dict, list, str, bytes, or any pickle-able object)
            path: File path to save to
            **kwargs: 
                - format: 'json', 'text', 'binary', 'pickle' (auto-detected if not specified)
                - encoding: Text encoding (default: 'utf-8')
                - ensure_dir: Create parent directories (default: True)
                - metadata: Dict of file metadata to store as xattr (if supported)
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_path = self._get_full_path(path)
            
            # Ensure parent directory exists
            if kwargs.get('ensure_dir', True):
                full_path.parent.mkdir(parents=True, exist_ok=True)

            format_type = kwargs.get('format', self._detect_format(data, full_path))
            encoding = kwargs.get('encoding', 'utf-8')

            if format_type == 'json':
                with open(full_path, 'w', encoding=encoding) as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format_type == 'text':
                with open(full_path, 'w', encoding=encoding) as f:
                    f.write(str(data))
            elif format_type == 'binary':
                with open(full_path, 'wb') as f:
                    if isinstance(data, bytes):
                        f.write(data)
                    else:
                        raise ValueError("Binary format requires bytes data")
            elif format_type == 'pickle':
                with open(full_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Store metadata if provided (attempt, but don't fail if not supported)
            if 'metadata' in kwargs and kwargs['metadata']:
                try:
                    self._store_file_metadata(full_path, kwargs['metadata'])
                except Exception as e:
                    self.logger.debug(f"Failed to store metadata for {full_path}: {e}")

            self.logger.info(f"Successfully saved data to: {full_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save data to {path}: {e}")
            return False

    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """
        Load data from local file system.
        
        Args:
            path: File path to load from
            **kwargs:
                - format: 'json', 'text', 'binary', 'pickle' (auto-detected if not specified)
                - encoding: Text encoding (default: 'utf-8')
                - default: Default value if file doesn't exist
                
        Returns:
            Any: Loaded data
        """
        try:
            full_path = self._get_full_path(path)
            
            if not full_path.exists():
                if 'default' in kwargs:
                    return kwargs['default']
                raise FileNotFoundError(f"File not found: {full_path}")

            format_type = kwargs.get('format', self._detect_format_from_extension(full_path))
            encoding = kwargs.get('encoding', 'utf-8')

            if format_type == 'json':
                with open(full_path, 'r', encoding=encoding) as f:
                    return json.load(f)
            elif format_type == 'text':
                with open(full_path, 'r', encoding=encoding) as f:
                    return f.read()
            elif format_type == 'binary':
                with open(full_path, 'rb') as f:
                    return f.read()
            elif format_type == 'pickle':
                with open(full_path, 'rb') as f:
                    return pickle.load(f)
            else:
                # Default to text
                with open(full_path, 'r', encoding=encoding) as f:
                    return f.read()

        except Exception as e:
            self.logger.error(f"Failed to load data from {path}: {e}")
            raise

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if file exists."""
        try:
            full_path = self._get_full_path(path)
            return full_path.exists()
        except Exception as e:
            self.logger.error(f"Error checking file existence for {path}: {e}")
            return False

    def delete(self, path: Union[str, Path]) -> bool:
        """Delete file or directory."""
        try:
            full_path = self._get_full_path(path)
            
            if full_path.is_file():
                full_path.unlink()
                self.logger.info(f"Deleted file: {full_path}")
            elif full_path.is_dir():
                shutil.rmtree(full_path)
                self.logger.info(f"Deleted directory: {full_path}")
            else:
                self.logger.warning(f"Path does not exist: {full_path}")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete {path}: {e}")
            return False

    def list_files(self, path: Union[str, Path], pattern: Optional[str] = None) -> List[Path]:
        """List files in directory."""
        try:
            full_path = self._get_full_path(path)
            
            if not full_path.is_dir():
                return []
            
            if pattern:
                files = list(full_path.glob(pattern))
            else:
                files = [f for f in full_path.iterdir() if f.is_file()]
            
            return sorted(files)

        except Exception as e:
            self.logger.error(f"Failed to list files in {path}: {e}")
            return []

    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy file from source to destination."""
        try:
            src_path = self._get_full_path(src)
            dst_path = self._get_full_path(dst)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_path, dst_path)
            self.logger.info(f"Copied file: {src_path} -> {dst_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to copy file {src} to {dst}: {e}")
            return False

    def move_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Move file from source to destination."""
        try:
            src_path = self._get_full_path(src)
            dst_path = self._get_full_path(dst)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src_path), str(dst_path))
            self.logger.info(f"Moved file: {src_path} -> {dst_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to move file {src} to {dst}: {e}")
            return False

    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        try:
            full_path = self._get_full_path(path)
            return full_path.stat().st_size
        except Exception as e:
            self.logger.error(f"Failed to get file size for {path}: {e}")
            return -1

    def _detect_format(self, data: Any, path: Path) -> str:
        """Detect format based on data type and file extension."""
        extension = path.suffix.lower()
        
        if extension == '.json' or isinstance(data, (dict, list)):
            return 'json'
        elif extension in ['.txt', '.md', '.csv'] or isinstance(data, str):
            return 'text'
        elif extension in ['.pkl', '.pickle']:
            return 'pickle'
        elif isinstance(data, bytes):
            return 'binary'
        else:
            return 'text'  # Default

    def _detect_format_from_extension(self, path: Path) -> str:
        """Detect format from file extension."""
        extension = path.suffix.lower()
        
        if extension == '.json':
            return 'json'
        elif extension in ['.pkl', '.pickle']:
            return 'pickle'
        elif extension in ['.bin', '.dat']:
            return 'binary'
        else:
            return 'text'  # Default

    def _store_file_metadata(self, path: Path, metadata: dict) -> None:
        """Store file metadata as extended attributes or companion file."""
        try:
            # Try to use extended attributes first (Unix/Linux)
            import os
            if hasattr(os, 'setxattr'):
                for key, value in metadata.items():
                    try:
                        os.setxattr(str(path), f'user.{key}'.encode(), str(value).encode())
                    except (OSError, AttributeError):
                        continue
                return
        except ImportError:
            pass
        
        # Fallback: create companion metadata file
        try:
            metadata_path = path.with_suffix(path.suffix + '.meta')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Failed to create metadata file: {e}")

    def get_file_metadata(self, path: Union[str, Path]) -> Optional[dict]:
        """Get file metadata from extended attributes or companion file."""
        full_path = self._get_full_path(path)
        metadata = {}
        
        try:
            # Try extended attributes first
            import os
            if hasattr(os, 'listxattr') and hasattr(os, 'getxattr'):
                try:
                    for attr in os.listxattr(str(full_path)):
                        if attr.startswith('user.'):
                            key = attr[5:]  # Remove 'user.' prefix
                            value = os.getxattr(str(full_path), attr).decode()
                            metadata[key] = value
                    if metadata:
                        return metadata
                except (OSError, AttributeError):
                    pass
        except ImportError:
            pass
        
        # Fallback: check for companion metadata file
        try:
            metadata_path = full_path.with_suffix(full_path.suffix + '.meta')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return None
