import os
import weakref
from . import _sharing_strategy


_shared_cache = weakref.WeakValueDictionary()


def _fd_id(fd):
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def _shm_object_handle(handle):
    if len(handle) == 3:
        return handle[1]
    else:
        return _fd_id(handle[0])


def _shared_serialize(self):
    handle, weak_storage = self._share(_sharing_strategy == 'file_descriptor')
    object_handle = _shm_object_handle(handle)
    _shared_cache[object_handle] = weak_storage
    return handle


@classmethod
def _shared_deserialize(cls, args):
    object_handle = _shm_object_handle(args)

    try:
        weak_storage = _shared_cache[object_handle]
        # Try to momentarily convert a weak reference into a strong one
        weak_storage.retain()
        if weak_storage._cdata != 0:
            # Success, we managed to retain the storage before it was freed
            new_storage = type(weak_storage)(cdata=weak_storage._cdata)
            return new_storage
    except KeyError:
        pass

    new_storage, weak_storage = cls._new_shared(*args)
    _shared_cache[object_handle] = weak_storage
    return new_storage


@classmethod
def _save_shared_args(cls, args):
    storage = cls()
    storage._shared_args = args
    storage._tensor_users = set()
    return storage


def _open_shared_fd(self, fd_map):
    shared_args = (fd_map[self._shared_args[0]], self._shared_args[1])
    storage = self._shared_deserialize(shared_args)
    self._set_cdata(storage._cdata)
    for t in self._tensor_users:
        t.set_(storage, t.storageOffset(), t.size(), t.stride())
    del self._shared_args
    del self._tensor_users


def _init_storage_sharing():
    from torch.Storage import _StorageBase
    _StorageBase._shared_serialize = _shared_serialize
    _StorageBase._shared_deserialize = _shared_deserialize
    _StorageBase._save_shared_args = _save_shared_args
    _StorageBase._open_shared_fd = _open_shared_fd

