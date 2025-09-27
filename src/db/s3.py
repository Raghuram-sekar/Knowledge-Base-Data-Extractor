import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from threading import Lock
from typing import Optional, List
from core.config import config
from core.logger import logger


class S3Client:
    """Singleton S3 client for interacting with S3-compatible storage."""
    
    _instance: Optional['S3Client'] = None
    _lock = Lock()  # thread-safe singleton

    def __new__(cls, endpoint_url: Optional[str] = None, access_key: Optional[str] = None, 
                secret_key: Optional[str] = None, region: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(S3Client, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, endpoint_url: Optional[str] = None, access_key: Optional[str] = None, 
                 secret_key: Optional[str] = None, region: Optional[str] = None):
        if self._initialized:
            return
            
        # Use config values as defaults
        self.endpoint_url = endpoint_url or config.S3_ENDPOINT_URL
        self.access_key = access_key or config.S3_ACCESS_KEY
        self.secret_key = secret_key or config.S3_SECRET_KEY
        self.region = region or config.S3_REGION or "us-east-1"
        self.logger = logger.get_logger()
        
        if not all([self.access_key, self.secret_key]):
            raise ValueError("S3 access key and secret key are required")
        
        self._initialize_client()
        self._initialized = True

    def _initialize_client(self):
        """Initialize the boto3 S3 client."""
        try:
            client_config = Config(signature_version="s3v4")
            
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=client_config,
                region_name=self.region,
            )
            
            # Test connection
            self._s3.list_buckets()
            self.logger.info(f"S3 client initialized successfully with endpoint: {self.endpoint_url}")
            
        except NoCredentialsError:
            self.logger.error("S3 credentials not found")
            raise
        except ClientError as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error initializing S3 client: {e}")
            raise

    @property
    def client(self):
        """Get the boto3 S3 client."""
        return self._s3

    # Bucket operations
    def create_bucket(self, bucket_name: str) -> bool:
        """Create an S3 bucket."""
        try:
            self._s3.head_bucket(Bucket=bucket_name)
            self.logger.info(f"Bucket '{bucket_name}' already exists.")
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    # Create bucket
                    if self.region != 'us-east-1':
                        self._s3.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    else:
                        self._s3.create_bucket(Bucket=bucket_name)
                    
                    self.logger.info(f"Bucket '{bucket_name}' created successfully.")
                    return True
                except ClientError as ce:
                    self.logger.error(f"Failed to create bucket '{bucket_name}': {ce}")
                    return False
            else:
                self.logger.error(f"Error checking bucket '{bucket_name}': {e}")
                return False

    def list_buckets(self) -> List[str]:
        """List all S3 buckets."""
        try:
            response = self._s3.list_buckets()
            buckets = [b["Name"] for b in response["Buckets"]]
            self.logger.debug(f"Found {len(buckets)} buckets")
            return buckets
        except ClientError as e:
            self.logger.error(f"Failed to list buckets: {e}")
            return []

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if bucket exists."""
        try:
            self._s3.head_bucket(Bucket=bucket_name)
            return True
        except ClientError:
            return False

    # Object operations
    def upload_file(self, bucket_name: str, file_path: str, object_name: str, 
                   extra_args: Optional[dict] = None) -> bool:
        """Upload a file to S3."""
        try:
            self._s3.upload_file(file_path, bucket_name, object_name, ExtraArgs=extra_args)
            self.logger.info(f"Uploaded {file_path} → s3://{bucket_name}/{object_name}")
            return True
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return False
        except ClientError as e:
            self.logger.error(f"Failed to upload file: {e}")
            return False

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """Download a file from S3."""
        try:
            self._s3.download_file(bucket_name, object_name, file_path)
            self.logger.info(f"Downloaded s3://{bucket_name}/{object_name} → {file_path}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to download file: {e}")
            return False

    def upload_fileobj(self, bucket_name: str, file_obj, object_name: str, 
                      extra_args: Optional[dict] = None) -> bool:
        """Upload a file-like object to S3."""
        try:
            self._s3.upload_fileobj(file_obj, bucket_name, object_name, ExtraArgs=extra_args)
            self.logger.info(f"Uploaded file object → s3://{bucket_name}/{object_name}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to upload file object: {e}")
            return False

    def download_fileobj(self, bucket_name: str, object_name: str, file_obj) -> bool:
        """Download a file from S3 to a file-like object."""
        try:
            self._s3.download_fileobj(bucket_name, object_name, file_obj)
            self.logger.info(f"Downloaded s3://{bucket_name}/{object_name} to file object")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to download to file object: {e}")
            return False

    def list_objects(self, bucket_name: str, prefix: str = "") -> List[str]:
        """List objects in S3 bucket."""
        try:
            response = self._s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            objects = [obj["Key"] for obj in response.get("Contents", [])]
            self.logger.debug(f"Found {len(objects)} objects in bucket '{bucket_name}' with prefix '{prefix}'")
            return objects
        except ClientError as e:
            self.logger.error(f"Failed to list objects: {e}")
            return []

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """Check if object exists in S3."""
        try:
            self._s3.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError:
            return False

    def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """Delete an object from S3."""
        try:
            self._s3.delete_object(Bucket=bucket_name, Key=object_name)
            self.logger.info(f"Deleted s3://{bucket_name}/{object_name}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to delete object: {e}")
            return False

    def get_object_info(self, bucket_name: str, object_name: str) -> Optional[dict]:
        """Get object metadata."""
        try:
            response = self._s3.head_object(Bucket=bucket_name, Key=object_name)
            return {
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag')
            }
        except ClientError as e:
            self.logger.error(f"Failed to get object info: {e}")
            return None


# Factory function to get S3Client instance
def get_s3_client(endpoint_url: Optional[str] = None, access_key: Optional[str] = None,
                  secret_key: Optional[str] = None, region: Optional[str] = None) -> S3Client:
    """Get S3Client singleton instance."""
    return S3Client(endpoint_url, access_key, secret_key, region)
