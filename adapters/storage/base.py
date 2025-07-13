from abc import ABC, abstractmethod
from typing import Optional, Union, BinaryIO, Iterator, List, Dict, Protocol
from pathlib import Path
from contextlib import contextmanager
from adapters.types import StorageLocation, StorageObject


class ReadableStorageAdapter(ABC):
    """Interface for reading from a storage system."""

    @abstractmethod
    def exists(self, location: Union[str, StorageLocation]) -> bool:
        """Check if an object exists."""
        pass

    @abstractmethod
    def get_object(self, location: Union[str, StorageLocation]) -> bytes:
        """Get object content as bytes."""
        pass

    @abstractmethod
    def get_object_stream(self, location: Union[str, StorageLocation]) -> BinaryIO:
        """Get object as a binary stream for reading."""
        pass

    @abstractmethod
    @contextmanager
    def get_object_as_file(
            self,
            location: Union[str, StorageLocation],
            suffix: Optional[str] = None
    ) -> Iterator[Path]:
        """
        Get object as a temporary file.

        Args:
            location: Storage location
            suffix: File suffix (e.g., '.jpg', '.wav')

        Yields:
            Path to temporary file (deleted after context)
        """
        pass

    @abstractmethod
    def list_objects(
            self,
            prefix: str,
            bucket: Optional[str] = None,
            max_keys: Optional[int] = None
    ) -> List[StorageObject]:
        """
        List objects with a given prefix.

        Args:
            prefix: Object key prefix to filter by
            bucket: Bucket name (uses default if not specified)
            max_keys: Maximum number of objects to return

        Returns:
            List of StorageObject metadata
        """
        pass

    @abstractmethod
    def get_presigned_url(
            self,
            location: Union[str, StorageLocation],
            expiration: int = 3600,
            method: str = "GET"
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            location: Storage location
            expiration: URL expiration time in seconds
            method: HTTP method ('GET', 'PUT', 'DELETE')

        Returns:
            Presigned URL string
        """
        pass

    @abstractmethod
    def parse_url(self, url: str) -> StorageLocation:
        """
        Parse a storage URL into its components.

        Args:
            url: Storage URL (e.g., 's3://bucket/key')

        Returns:
            StorageLocation with parsed components
        """
        pass


class WritableStorageAdapter(ABC):
    """Interface for writing to a storage system."""

    @abstractmethod
    def put_object(
            self,
            location: Union[str, StorageLocation],
            data: Union[bytes, BinaryIO, Path],
            content_type: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None
    ) -> StorageObject:
        """
        Upload an object to storage.

        Args:
            location: Storage location
            data: Object data (bytes, file-like object, or file path)
            content_type: MIME type of the object
            metadata: Additional metadata to store with object

        Returns:
            StorageObject with upload details
        """
        pass

    @abstractmethod
    def delete_object(self, location: Union[str, StorageLocation]) -> None:
        """
        Delete an object from storage.

        Args:
            location: Storage location

        Raises:
            FileNotFoundError: If object doesn't exist
        """
        pass

    @abstractmethod
    def put_object_from_file(
            self,
            location: Union[str, StorageLocation],
            file_path: Union[str, Path],
            content_type: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None
    ) -> StorageObject:
        """
        Upload a file to storage.

        Args:
            location: Storage location
            file_path: Path to local file
            content_type: MIME type (auto-detected if not provided)
            metadata: Additional metadata

        Returns:
            StorageObject with upload details
        """
        pass

    @abstractmethod
    def copy_object(
            self,
            source: Union[str, StorageLocation],
            destination: Union[str, StorageLocation],
            metadata: Optional[Dict[str, str]] = None
    ) -> StorageObject:
        """
        Copy an object within storage.

        Args:
            source: Source location
            destination: Destination location
            metadata: New metadata (or copy existing)

        Returns:
            StorageObject for the copy
        """
        pass


class StorageAdapter(ReadableStorageAdapter, WritableStorageAdapter, ABC):
    """
    Full storage adapter with both read and write capabilities.

    This combines both interfaces for adapters that support all operations.
    Concrete implementations should inherit from this for full functionality.
    """
    pass


class SupportsGetObject(Protocol):
    """Protocol for classes that have get_object method."""

    def get_object(self, location: Union[str, StorageLocation]) -> bytes: ...


class SupportsDeleteObject(Protocol):
    """Protocol for classes that have delete_object method."""

    def delete_object(self, location: Union[str, StorageLocation]) -> None: ...


class BatchOperationsMixin:
    """
    Mixin for storage adapters that support batch operations.

    This mixin expects to be combined with classes that implement
    the required methods.
    """

    def get_objects_batch(
            self: SupportsGetObject,
            locations: List[Union[str, StorageLocation]]
    ) -> Dict[str, Union[bytes, Exception]]:
        """Get multiple objects in parallel."""
        results = {}
        for location in locations:
            try:
                results[str(location)] = self.get_object(location)
            except Exception as e:
                results[str(location)] = e
        return results

    def delete_objects_batch(
            self: SupportsDeleteObject,
            locations: List[Union[str, StorageLocation]]
    ) -> Dict[str, Optional[Exception]]:
        """Delete multiple objects."""
        results = {}
        for location in locations:
            try:
                self.delete_object(location)
                results[str(location)] = None
            except Exception as e:
                results[str(location)] = e
        return results


# Optional: Async support interface
class AsyncReadableStorageAdapter(ABC):
    """Async version of readable storage adapter."""

    @abstractmethod
    async def exists(self, location: Union[str, StorageLocation]) -> bool:
        pass

    @abstractmethod
    async def get_object(self, location: Union[str, StorageLocation]) -> bytes:
        pass
