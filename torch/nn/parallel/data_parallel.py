import weakref
import torch
from torch.nn.modules import Container
import torch.multiprocessing as mp
from . import parallel
import os


def _worker_loop(model, inqueue, outqueue, device):
    outputs = {}
    functions = {}
    inputs = {}

    def do_forward(input):
        output = model(input)
        var_id = id(output)
        outputs[var_id] = output
        info = {
            'data': output.data,
            'id': var_id,
            'volatile': output.volatile,
            'requires_grad': output.requires_grad
        }
        if output.creator is not None:
            fn_id = id(output.creator)
            info['function_id'] = fn_id
            functions[fn_id] = output.creator
            inputs[fn_id] = input
        return info

    def do_backward(fn_id, grad):
        tmp = torch.autograd.Variable(torch.Tensor(), functions[fn_id],
                                      requires_grad=True)
        tmp.backward(grad)
        input_var = inputs[fn_id]
        return input_var.grad

    while True:
        msg, data = inqueue.get()
        print('msg', msg)
        if msg == 'quit':
            break
        elif msg == 'model':
            model = data
        elif msg == 'forward':
            outqueue.put(do_forward(torch.autograd.Variable(**data)))
        elif msg == 'backward':
            outqueue.put(do_backward(*data))
        elif msg == 'del variable':
            del outputs[data]
        elif msg == 'del function':
            del functions[data]
            del inputs[data]


class DataParallel(Container):
    def __init__(self, module, devices, output_device=None):
        super(DataParallel, self).__init__()
        self.devices = devices
        self.output_device = output_device
        modules = self.replicate(module, devices)
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.inqueues, self.outqueues, self.processes = self._create_processes()
        for p in self.processes:
            p.start()

    def __del__(self):
        for queue in self.inqueues:
            queue.put(('quit', None))

    def forward(self, input):
        output_device = self.output_device
        if output_device is None:
            output_device = -1 if not input.is_cuda else input.get_device()

        inputs = self.scatter(input)
        outputs = self.parallel_apply(inputs)
        print('parallel_apply done')
        return self.gather(outputs, output_device)

    def parallel_apply(self, inputs):
        return ParallelApply(self.inqueues, self.outqueues)(*inputs)

    def scatter(self, input):
        return parallel.scatter(input, self.devices)

    def gather(self, outputs, output_device):
        return parallel.gather(outputs, output_device)

    def replicate(self, module, devices):
        return parallel.replicate(module, devices, unlink=True)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        return self._modules[str(idx)]

    def _create_processes(self):
        inqueues = []
        outqueues = []
        processes = []
        for i, m in enumerate(self.children()):
            print('Creating subprocess', i)
            inqueue = mp.Queue()
            outqueue = mp.Queue()
            p = mp.Process(target=_worker_loop, args=(None, inqueue, outqueue, i))
            inqueue.put(('model', m))
            inqueues.append(inqueue)
            outqueues.append(outqueue)
            processes.append(p)
        return inqueues, outqueues, processes


class ParallelApply(torch.autograd.Function):
    def __init__(self, inqueues, outqueues):
        self.inqueues = inqueues
        self.outqueues = outqueues
        self.refs = []
        self.var_ids = []
        self.function_ids = []

    def __del__(self):
        for function_id, queue in zip(self.function_ids, self.inqueues):
            if function_id:
                queue.put(('del function', function_id))

    def _do_forward(self, *args):
        output = super(ParallelApply, self)._do_forward(*args)

        def make_destructor(queue, var_id):
            def del_variable(ref):
                print("deleting", ref, var_id)
                queue.put(('del variable', var_id))
            return del_variable

        for var, var_id, queue in zip(output, self.var_ids, self.inqueues):
            print('make_destructor', format(id(var), '#x'), format(var_id, '#x'))
            self.refs.append(weakref.ref(var, make_destructor(queue, var_id)))
        return output

    def forward(self, *inputs):
        for input, queue in zip(inputs, self.inqueues):
            queue.put(('forward', {
                'data': input,
                'volatile': self.volatile,
                'requires_grad': self.requires_grad,
            }))

        outputs = []
        for queue in self.outqueues:
            print('reading from outqueue...')
            info = queue.get()
            self.function_ids.append(info.get('function_id', None))
            self.var_ids.append(info['id'])
            outputs.append(info['data'])
            if not info['requires_grad']:
                self.mark_non_differentiable(info['data'])

        return tuple(outputs)

    __call__ = _do_forward

    def backward(self, *grads):
        print('calling backward...')
        for fn_id, grad, queue in zip(self.function_ids, grads, self.inqueues):
            queue.put(('backward', (fn_id, grad)))
        grad_inputs = []
        for queue in self.outqueues:
            grad_inputs.append(queue.get())
        print('gots the grad inputs')
        return tuple(grad_inputs)
