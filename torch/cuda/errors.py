from . import cudart


SUCCESS = 0
ERROR_NOT_READY = 34


class CudaError(RuntimeError):
    def __init__(self, code):
        msg = cudart().cudaGetErrorString(code).decode('utf-8')
        super(CudaError, self).__init__('{0} ({1})'.format(msg, code))


def check_error(res):
    if res != SUCCESS:
        raise CudaError(res)
