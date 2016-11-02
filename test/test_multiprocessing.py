import contextlib
import gc
import os
import sys
import time
import unittest
from sys import platform

import torch
import torch.cuda
import torch.multiprocessing as mp
from common import TestCase


HAS_SHM_FILES = os.path.isdir('/dev/shm')
TEST_CUDA_IPC = torch.cuda.is_available() and sys.version_info[0] == 3


def simple_fill(queue, event):
    data = queue.get()
    data[0][:] = 4
    event.set()


def simple_pool_fill(tensor):
    tensor.fill_(4)
    return tensor.add(1)


# Multiply by two in a separate stream
def cuda_multiply_two(queue, ready, done):
    ready.set()
    with torch.cuda.stream(torch.cuda.Stream()):
        cuda_event, tensor = queue.get()
        cuda_event.wait()
        tensor.mul_(2)
        cuda_event.record()
        done.set()
        del cuda_event


@contextlib.contextmanager
def fs_sharing():
    prev_strategy = mp.get_sharing_strategy()
    mp.set_sharing_strategy('file_system')
    try:
        yield
    finally:
        mp.set_sharing_strategy(prev_strategy)


class leak_checker(object):

    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        self.next_fd = self._get_next_fd()
        return self

    def __exit__(self, *args):
        if args[0] is None:
            gc.collect()
            self.test_case.assertEqual(self.next_fd, self._get_next_fd())
            self.test_case.assertFalse(self.has_shm_files())
        return False

    def check_pid(self, pid):
        self.checked_pids.append(pid)

    def _get_next_fd(self):
        # dup uses the lowest-numbered unused descriptor for the new descriptor
        fd = os.dup(0)
        os.close(fd)
        return fd

    def has_shm_files(self, wait=True):
        if not HAS_SHM_FILES:
            return False
        result = self._has_shm_files()
        if result and mp.get_sharing_strategy() == 'file_system' and wait:
            time.sleep(0.5)
            return self._has_shm_files()
        return result

    def _has_shm_files(self):
        gc.collect()
        names = list('torch_' + str(pid) for pid in self.checked_pids)
        for filename in os.listdir('/dev/shm'):
            for name in names:
                if filename.startswith(name):
                    return True
        return False


class TestMultiprocessing(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMultiprocessing, self).__init__(*args, **kwargs)

    def _test_sharing(self, type=torch.FloatTensor):
        def do_test():
            x = torch.zeros(5, 5).type(type)
            q = mp.Queue()
            e = mp.Event()
            data = [x, x[:, 1]]
            q.put(data)
            p = mp.Process(target=simple_fill, args=(q, e))
            lc.check_pid(p.pid)
            p.start()
            e.wait()
            self.assertTrue(data[0].eq(4).all())
            self.assertTrue(data[1].eq(4).all())
            p.join(1)
            self.assertFalse(p.is_alive())

        with leak_checker(self) as lc:
            do_test()

    def _test_preserve_sharing(self):
        def do_test():
            x = torch.randn(5, 5)
            data = [x.storage(), x.storage()[1:4], x, x[2], x[:,1]]
            q = mp.Queue()
            q.put(data)
            new_data = q.get()
            self.assertEqual(new_data, data, 0)
            storage_cdata = data[0]._cdata
            self.assertEqual(new_data[0]._cdata, storage_cdata)
            for t in new_data[2:]:
                self.assertEqual(t.storage()._cdata, storage_cdata)
            # TODO: enable after fixing #46
            # new_data[0].fill_(10)
            # self.assertEqual(new_data[1], new_data[0][1:4], 0)

        with leak_checker(self):
            do_test()

    def _test_pool(self):
        def do_test():
            p = mp.Pool(2)
            for proc in p._pool:
                lc.check_pid(proc.pid)

            buffers = (torch.zeros(2, 2) for i in range(4))
            results = p.map(simple_pool_fill, buffers, 1)
            for r in results:
                self.assertEqual(r, torch.ones(2, 2) * 5, 0)
            self.assertEqual(len(results), 4)

            p.close()
            p.join()

        with leak_checker(self) as lc:
            do_test()

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_fd_sharing(self):
        self._test_sharing()

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_fd_preserve_sharing(self):
        self._test_preserve_sharing()

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_fd_pool(self):
        self._test_pool()

    def test_fs_sharing(self):
        with fs_sharing():
            self._test_sharing()

    def test_fs_preserve_sharing(self):
        with fs_sharing():
            self._test_preserve_sharing()

    def test_fs_pool(self):
        with fs_sharing():
            self._test_pool()

    @unittest.skipIf(not HAS_SHM_FILES, "don't not how to check if shm files exist")
    def test_fs(self):
        with fs_sharing(), leak_checker(self) as lc:
            x = torch.DoubleStorage(4)
            q = mp.Queue()
            self.assertFalse(lc.has_shm_files())
            q.put(x)
            self.assertTrue(lc.has_shm_files(wait=False))
            q.get()
            del x
            del q  # We have to clean up fds for leak_checker

    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_cuda(self):
        self._test_sharing(torch.cuda.FloatTensor)

    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_event(self):
        queue = mp.Queue()
        ready = mp.Event()
        done = mp.Event()
        p = mp.Process(target=cuda_multiply_two, args=(queue, ready, done))
        p.start()

        ready.wait()
        with torch.cuda.stream(torch.cuda.Stream()):
            tensor = torch.cuda.FloatTensor([1, 1, 1, 1])
            # Use a sleep kernel to test events. Without the event, the
            # multiply happens before the add.
            event = torch.cuda.Event(interprocess=True)
            torch.cuda.sleep(20000000)  # about 30 ms
            tensor.add_(1)
            event.record()
            queue.put((event, tensor))
            done.wait()  # must wait until subprocess records event
            event.synchronize()
            self.assertEqual(list(tensor), [4, 4, 4, 4])
        p.join()

if __name__ == '__main__':
    unittest.main()
