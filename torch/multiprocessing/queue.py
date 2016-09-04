import os
import socket
import multiprocessing
from itertools import chain
from io import BytesIO

import torch
from .common import CustomizablePicklingQueue, ExtendedInitPickler, ExtendedInitUnpickler, reduce_torch_object


class Queue(CustomizablePicklingQueue):
    def __init__(self, context=None, reducers=None):
        if context is None:
            context = multiprocessing.get_context()
        if reducers is None:
            reducers = {}

        for t in chain(torch._tensor_classes, torch._storage_classes):
            reducers[t] = reduce_torch_object

        super(Queue, self).__init__(context, reducers)


class FdQueue(Queue):

    def __init__(self, *args, **kwargs):
        super(FdQueue, self).__init__(*args, **kwargs)
        self._fd_reader, self._fd_writer = socket.socketpair(socket.AF_UNIX)

    def _send_reducers(self, obj):
        buffer = BytesIO()
        pickler = ExtendedInitPickler(buffer, self._reducers)
        pickler.dump(obj)
        # We need a list of unique file descriptors
        fd_list = list(set(obj._get_shared_fd() for obj in pickler.extended_init))
        pickler.dump(fd_list)
        socket_fd = self._fd_writer.fileno()
        for fd in fd_list:
            torch._C._sendfd(socket_fd, fd)
        self._writer.send_bytes(buffer.getvalue())

    def _load(self):
        buf = BytesIO(self._reader.recv_bytes())
        pickler = ExtendedInitUnpickler(buf)
        result = pickler.load()
        fd_list = pickler.load()
        socket_fd = self._fd_reader.fileno()
        fd_map = {fd: torch._C._recvfd(socket_fd) for fd in fd_list}
        for obj in pickler.extended_init:
            obj._open_shared_fd(fd_map)
        for new_fd in fd_map.values():
            os.close(new_fd)
        return result

