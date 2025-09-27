from .storage import Storage
from pathlib import Path
from typing import Any, Union, Optional, List
import json
import pickle
import io
import re
from core.config import config
from core.logger import logger
from db.s3 import get_s3_client, S3Client


class S3Storage(Storage):
    """S3-compatible storage implementation."""

    def __init__(self, bucket_name: Optional[str] = None, prefix: str = "", 
                 endpoint_url: Optional[str] = None, access_key: Optional[str] = None,
                 secret_key: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize S3Storage.
        
        Args:
            bucket_name: S3 bucket name (uses config default if not provided)
            prefix: Prefix for all object keys
            endpoint_url: S3 endpoint URL (uses config default if not provided)
            access_key: S3 access key (uses config default if not provided)
            secret_key: S3 secret key (uses config default if not provided)
            region: S3 region (uses config default if not provided)
        """
        super().__init__()
        
        # Use config values as defaults
        self.bucket_name = bucket_name or config.S3_BUCKET_NAME
        self.prefix = prefix.rstrip('/') + '/' if prefix else ""
        self.logger = logger.get_logger()
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name is required (either in config or as parameter)")
        
        # Initialize S3 client
        self.s3_client: S3Client = get_s3_client(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region
        )
        
        # Ensure bucket exists
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Ensure the S3 bucket exists."""
        try:
            if not self.s3_client.bucket_exists(self.bucket_name):
                self.s3_client.create_bucket(self.bucket_name)
                self.logger.info(f"Created S3 bucket: {self.bucket_name}")
            else:
                self.logger.info(f"S3 bucket exists: {self.bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to ensure bucket exists: {e}")
            raise

    def _get_object_key(self, path: Union[str, Path]) -> str:
        """Get the full S3 object key with prefix."""
        path_str = str(path).lstrip('/')
        return f"{self.prefix}{path_str}"

    def save(self, data: Any, path: Union[str, Path], **kwargs) -> bool:
        """
        Save data to S3.
        
        Args:
            data: Data to save (dict, list, str, bytes, or any pickle-able object)
            path: S3 object key path
            **kwargs: 
                - format: 'json', 'text', 'binary', 'pickle' (auto-detected if not specified)
                - encoding: Text encoding (default: 'utf-8')
                - content_type: Content type for the object
                - metadata: Dict of metadata to store with object
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            object_key = self._get_object_key(path)
            format_type = kwargs.get('format', self._detect_format(data, Path(str(path))))
            encoding = kwargs.get('encoding', 'utf-8')
            
            # Prepare file object and extra args
            extra_args = {}
            
            if 'content_type' in kwargs:
                extra_args['ContentType'] = kwargs['content_type']
            elif format_type == 'json':
                extra_args['ContentType'] = 'application/json'
            elif format_type == 'text':
                extra_args['ContentType'] = 'text/plain'
                
            if 'metadata' in kwargs:
                # S3 metadata keys must be lowercase and ASCII
                s3_metadata = {}
                for key, value in kwargs['metadata'].items():
                    # Clean key: lowercase, replace invalid chars with hyphens
                    clean_key = re.sub(r'[^a-z0-9\-]', '-', str(key).lower())
                    # Ensure value is string and ASCII
                    clean_value = str(value).encode('ascii', errors='ignore').decode('ascii')
                    s3_metadata[clean_key] = clean_value
                extra_args['Metadata'] = s3_metadata

            # Convert data to appropriate format and upload
            if format_type == 'json':
                json_str = json.dumps(data, indent=2, ensure_ascii=False)
                file_obj = io.BytesIO(json_str.encode(encoding))
            elif format_type == 'text':
                file_obj = io.BytesIO(str(data).encode(encoding))
            elif format_type == 'binary':
                if isinstance(data, bytes):
                    file_obj = io.BytesIO(data)
                else:
                    raise ValueError("Binary format requires bytes data")
            elif format_type == 'pickle':
                pickle_data = pickle.dumps(data)
                file_obj = io.BytesIO(pickle_data)
                extra_args['ContentType'] = 'application/octet-stream'
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            # Upload to S3
            success = self.s3_client.upload_fileobj(
                self.bucket_name, 
                file_obj, 
                object_key, 
                extra_args=extra_args if extra_args else None
            )
            
            if success:
                self.logger.info(f"Successfully saved data to S3: s3://{self.bucket_name}/{object_key}")
            
            return success

        except Exception as e:
            self.logger.error(f"Failed to save data to S3 {path}: {e}")
            return False

    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """
        Load data from S3.
        
        Args:
            path: S3 object key path
            **kwargs:
                - format: 'json', 'text', 'binary', 'pickle' (auto-detected if not specified)
                - encoding: Text encoding (default: 'utf-8')
                - default: Default value if object doesn't exist
                
        Returns:
            Any: Loaded data
        """
        try:
            object_key = self._get_object_key(path)
            
            if not self.exists(path):
                if 'default' in kwargs:
                    return kwargs['default']
                raise FileNotFoundError(f"S3 object not found: s3://{self.bucket_name}/{object_key}")

            format_type = kwargs.get('format', self._detect_format_from_extension(Path(str(path))))
            encoding = kwargs.get('encoding', 'utf-8')

            # Download data
            file_obj = io.BytesIO()
            success = self.s3_client.download_fileobj(self.bucket_name, object_key, file_obj)
            
            if not success:
                raise Exception(f"Failed to download from S3: s3://{self.bucket_name}/{object_key}")
            
            file_obj.seek(0)

            # Parse based on format
            if format_type == 'json':
                content = file_obj.read().decode(encoding)
                return json.loads(content)
            elif format_type == 'text':
                return file_obj.read().decode(encoding)
            elif format_type == 'binary':
                return file_obj.read()
            elif format_type == 'pickle':
                return pickle.load(file_obj)
            else:
                # Default to text
                return file_obj.read().decode(encoding)

        except Exception as e:
            self.logger.error(f"Failed to load data from S3 {path}: {e}")
            raise

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if S3 object exists."""
        try:
            object_key = self._get_object_key(path)
            return self.s3_client.object_exists(self.bucket_name, object_key)
        except Exception as e:
            self.logger.error(f"Error checking S3 object existence for {path}: {e}")
            return False

    def delete(self, path: Union[str, Path]) -> bool:
        """Delete S3 object."""
        try:
            object_key = self._get_object_key(path)
            
            if not self.exists(path):
                self.logger.warning(f"S3 object does not exist: s3://{self.bucket_name}/{object_key}")
                return False
            
            success = self.s3_client.delete_object(self.bucket_name, object_key)
            
            if success:
                self.logger.info(f"Deleted S3 object: s3://{self.bucket_name}/{object_key}")
            
            return success

        except Exception as e:
            self.logger.error(f"Failed to delete S3 object {path}: {e}")
            return False

    def list_files(self, path: Union[str, Path], pattern: Optional[str] = None) -> List[str]:
        """List S3 objects with given prefix."""
        try:
            prefix = self._get_object_key(path)
            if not prefix.endswith('/'):
                prefix += '/'
                
            objects = self.s3_client.list_objects(self.bucket_name, prefix)
            
            # Filter by pattern if provided
            if pattern:
                import fnmatch
                objects = [obj for obj in objects if fnmatch.fnmatch(Path(obj).name, pattern)]
            
            # Remove prefix to get relative paths
            if self.prefix:
                objects = [obj[len(self.prefix):] for obj in objects if obj.startswith(self.prefix)]
            
            return sorted(objects)

        except Exception as e:
            self.logger.error(f"Failed to list S3 objects with prefix {path}: {e}")
            return []

    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy S3 object from source to destination."""
        try:
            src_key = self._get_object_key(src)
            dst_key = self._get_object_key(dst)
            
            copy_source = {'Bucket': self.bucket_name, 'Key': src_key}
            
            self.s3_client.client.copy(copy_source, self.bucket_name, dst_key)
            self.logger.info(f"Copied S3 object: s3://{self.bucket_name}/{src_key} -> s3://{self.bucket_name}/{dst_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to copy S3 object {src} to {dst}: {e}")
            return False

    def get_file_info(self, path: Union[str, Path]) -> Optional[dict]:
        """Get S3 object metadata."""
        try:
            object_key = self._get_object_key(path)
            return self.s3_client.get_object_info(self.bucket_name, object_key)
        except Exception as e:
            self.logger.error(f"Failed to get S3 object info for {path}: {e}")
            return None

    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get S3 object size in bytes."""
        try:
            info = self.get_file_info(path)
            return info['size'] if info else -1
        except Exception as e:
            self.logger.error(f"Failed to get S3 object size for {path}: {e}")
            return -1

    def generate_presigned_url(self, path: Union[str, Path], expiration: int = 3600) -> Optional[str]:
        """Generate a presigned URL for S3 object."""
        try:
            object_key = self._get_object_key(path)
            
            response = self.s3_client.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL for {path}: {e}")
            return None

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
