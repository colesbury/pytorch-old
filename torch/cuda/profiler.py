import ctypes
from . errors import check_error
from . import cudart


_output_modes = {
    'key-value': ctypes.c_int(0),
    'csv': ctypes.c_int(1),
}


def initialize(output_file, config_file='', output_mode='csv'):
    r"""Initializes the CUDA profiler

    Args:
        output_file (string): name of file where profiling results are stored
        config_file (string): name of configuration file
        output_mode (string): must be 'csv' or 'key-value'
    """
    if output_mode not in _output_modes:
        raise ValueError('invalid output_mode "{}" (valid modes are "{}")'
                         .format(output_mode, _output_modes.keys()))

    mode = _output_modes[output_mode]
    check_error(cudart().cudaProfilerInitialize(
        ctypes.c_char_p(config_file.encode('utf-8')),
        ctypes.c_char_p(output_file.encode('utf-8')),
        mode))


def start():
    r"""Starts or resumes the CUDA profiler.

    The program must be run under nvprof, the Visual Profiler, or the profiler
    must be initialized by calling `torch.cuda.profiler.initialize`.
    """
    check_error(cudart().cudaProfilerStart())


def stop():
    r"""Stops the CUDA profiler and flushes the output.
    """
    check_error(cudart().cudaProfilerStop())
