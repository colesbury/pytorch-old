import math
import unittest
import contextlib
from copy import deepcopy
from collections import OrderedDict

from common import make_jacobian, TestCase, iter_tensors, get_numerical_jacobian
from torch.autograd.functions import *
from torch.autograd import Variable

PRECISION = 1e-4

def iter_gradients(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield x.grad
    else:
        for elem in x:
            for result in iter_gradients(elem):
                yield result

def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()

def get_analytical_jacobian(input, output):
    jacobian = make_jacobian(input, output.numel())
    grad_output = output.data.clone().zero_()
    flat_grad_output = grad_output.view(-1)

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        zero_gradients(input)
        output.backward(grad_output, retain_variables=True)
        for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
            jacobian_x[:,i] = d_x

    return jacobian


@contextlib.contextmanager
def backward_engine(engine):
    _prev_engine = Variable._execution_engine
    Variable._execution_engine = engine()
    try:
        yield
    finally:
        Variable._execution_engine = _prev_engine

class TestAutograd(TestCase):

    def test_hooks(self):
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        counter = [0]
        def bw_hook(inc, grad):
            self.assertTrue(torch.is_tensor(grad))
            counter[0] += inc

        z = x ** 2 + x * 2 + x * y + y
        z.register_hook('test', lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(counter[0], 1)

        z.register_hook('test2', lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(counter[0], 4)

        z.remove_hook('test2')
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(counter[0], 5)

    def _test_backward(self):
        v_t = torch.randn(5, 5)
        x_t = torch.randn(5, 5)
        y_t = torch.rand(5, 5) + 0.1
        z_t = torch.randn(5, 5)
        grad_output = torch.randn(5, 5)
        v = Variable(v_t, requires_grad=True)
        x = Variable(x_t, requires_grad=True)
        y = Variable(y_t, requires_grad=True)
        z = Variable(z_t, requires_grad=True)

        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        a = x + (y * z) + 4 * z**2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z_t.pow(2) / y_t + 1
        y_grad = z_t - 4 * x_t * z_t.pow(2) / y_t.pow(2)
        z_grad = 8 * x_t * z_t / y_t + y_t
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_backward(self):
        self._test_backward()

    def test_backward_basic_engine(self):
        with backward_engine(torch.autograd.engine.BasicEngine):
            self._test_backward()

    def test_volatile(self):
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, volatile=True)

        z = x ** 2
        self.assertFalse(z.volatile)
        self.assertTrue(z.requires_grad)
        self.assertIsNotNone(z.creator)
        z.backward(torch.ones(5, 5))
        self.assertEqual(x.grad, torch.ones(5, 5) * 2)

        w = z + y
        self.assertTrue(w.volatile)
        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.creator)

    def test_indexing(self):
        x = torch.range(1, 16).resize_(4, 4)
        y = Variable(x)
        self.assertEqual(x[1], y[1].data)
        self.assertEqual(x[1, 1], y[1, 1].data[0])
        self.assertEqual(x[1:], y[1:].data)
        self.assertEqual(x[:2], y[:2].data)
        self.assertEqual(x[:2, 2], y[:2, 2].data)
        self.assertEqual(x[1:2, 2], y[1:2, 2].data)
        self.assertEqual(x[1, 2:], y[1, 2:].data)

    def test_requires_grad(self):
        x = Variable(torch.randn(5, 5))
        y = Variable(torch.randn(5, 5))
        z = Variable(torch.randn(5, 5), requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)
        def error():
            raise RuntimeError
        # Make sure backward isn't called on these
        a.backward_hooks = OrderedDict()
        x.backward_hooks = OrderedDict()
        y.backward_hooks = OrderedDict()
        a.backward_hooks['test'] = error
        x.backward_hooks['test'] = error
        y.backward_hooks['test'] = error
        b.backward(torch.ones(5, 5))

    def test_inplace(self):
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_variables=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_variables=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_variables=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(x.grad, torch.Tensor(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = Variable(torch.ones(5, 5), requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x.data, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    def test_shared_storage(self):
        x = Variable(torch.ones(5, 5))
        x_version = x._version
        y = x.t()
        z = x[1]
        self.assertEqual(x._version, x_version)
        z_version = z._version
        y.add_(2)
        self.assertNotEqual(x._version, x_version)
        self.assertNotEqual(z._version, z_version)

    def _test_setitem(self, size, index):
        x = Variable(torch.ones(*size), requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        if isinstance(index, Variable):
            index = index.data
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        x = Variable(torch.ones(*size), requires_grad=True)
        y = x + 2
        y_version = y._version
        value = Variable(torch.Tensor(x[index].size()).fill_(7), requires_grad=True)
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        if isinstance(index, Variable):
            index = index.data
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones(value.size()))

    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5,), 3)

    def test_setitem_mask(self):
        mask = torch.ByteTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_unused_output(self):
        x = Variable(torch.randn(10, 10), requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad, expected_grad)

        x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
            "CUDA not available or <2 GPUs detected")
    def test_unused_output_gpu(self):
        from torch.nn.parallel.functions import Broadcast
        x = Variable(torch.randn(5, 5).float().cuda(), requires_grad=True)
        outputs = Broadcast(list(range(torch.cuda.device_count())))(x)
        y = outputs[-1] * 2
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 2)

    def test_no_grad(self):
        x = Variable(torch.randn(10, 10), requires_grad=True)
        y = x + 2
        y = y.no_grad()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = Variable(torch.randn(10, 10), requires_grad=True)
        y = x * 2
        y = y.no_grad()
        self.assertFalse(y.requires_grad)
        self.assertFalse(y.creator.requires_grad)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. no_grad() is an advanced option.
        self.assertEqual(x.grad, torch.ones(10, 10))

    def test_type_conversions(self):
        import torch.cuda
        x = Variable(torch.randn(5, 5))
        self.assertIs(type(x.float().data), torch.FloatTensor)
        self.assertIs(type(x.int().data), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIs(type(x.float().cuda().data), torch.cuda.FloatTensor)
            self.assertIs(type(x.int().cuda().data), torch.cuda.IntTensor)
            self.assertIs(type(x.int().cuda().cpu().data), torch.IntTensor)
            if torch.cuda.device_count() > 2:
                x2 = x.float().cuda(1)
                self.assertIs(type(x2.data), torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIs(type(x2.data), torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIs(type(x2.data), torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)

    def test_backward_copy(self):
      # This tests checks backward engine for a very subtle bug that appreared
      # in one of the initial versions of autograd. Gradients tensors were
      # simply stored in lists while the function waited for all its gradients
      # to be computed. However, sometimes an output was used multiple times,
      # so the gradients needed to be summed. Engine used to keep a need_copy
      # set of tensors that will need a clone upon next addition and removed
      # them from the set as soon as the clone was performed. However, this
      # could lead to incorrect results if the same gradient tensor was
      # buffered in three places in the graph:
      # 1. When accumulating gradients in one of these places it was cloned
      #    and removed from need_copy set.
      # 2. When accumulating in second place, it wasn't in the need_copy set,
      #    so the gradients were simply accumulated in-place (which already
      #    modified the grad in 3rd place)
      # 3. When accumulating in the third place, it wasn't in the need_copy set
      #    as well, so the incoming gradient was summed in-place, yielding
      #    incorrect results in all functions, except the first one.
      x = Variable(torch.ones(5, 5), requires_grad=True)
      y = Variable(torch.ones(5, 5), requires_grad=True)
      # Simulate that we're in the middle of the graph
      a = x + 2
      b = y + 2
      c = x + 2
      # This op will just return grad_output two times in backward
      add1 = a + b
      add2 = add1 + c
      # Simulate a long branch, so grad_output will get buffered.
      for i in range(4):
        a = a * 2
        b = b * 2
        c = c * 2
      branch = a + b + c
      out = add2 + branch
      # expected gradients are:
      # for x: 34 (16 from final a, 16 from final c, 2 from add2)
      # for y: 17 (16 from final b, 1 from add2)
      grad_output = torch.ones(5, 5)
      out.backward(grad_output)
      self.assertEqual(x.grad, torch.ones(5, 5) * 34)
      self.assertEqual(y.grad, torch.ones(5, 5) * 17)


def index_variable(num_indices, max_indices):
    index = torch.randperm(max_indices)[:num_indices].long()
    return Variable(index, requires_grad=False)


L = 20
M = 10
S = 5
function_tests = [
    (Add,           (),                 ((M, M), (M, M))                            ),
    (Sub,           (),                 ((M, M), (M, M))                            ),
    (Mul,           (),                 ((M, M), (M, M))                            ),
    (Div,           (),                 ((M, M), torch.rand(M, M) + 5e-2)           ),
    (Pow,           (),                 (torch.rand(M, M) + 1e-3, torch.rand(M, M) + 0.1)),
    (AddConstant,   (3.14,),            ((L, L),)                                   ),
    (SubConstant,   (3.14,),            ((L, L),)                                   ),
    (SubConstant,   (3.14, True),       ((L, L),),                  'from_tensor'   ),
    (MulConstant,   (3.14,),            ((L, L),)                                   ),
    (DivConstant,   (3.14, True),       (torch.rand(L, L) + 1e-1,), 'by_tensor'     ),
    (PowConstant,   (3.14,),            (torch.rand(L, L),)                         ),
    (Transpose,     (0, 1),             (torch.rand(L, L),)                         ),
    (Transpose,     (2, 0),             (torch.rand(S, S, S),),     '3d'            ),
    (Permute,       (0, 4, 3, 5, 1, 2), ((1, 2, 3, 4, 5, 6),)                       ),
    (Index,         ((1, 2),),          (torch.rand(S, S, S),)                      ),
    (Index,         (slice(0, 3),),     (torch.rand(S, S, S),),     'slice'         ),
    (Index,         ((slice(0, 3), 1),),(torch.rand(S, S, S),),     'slice_index'   ),
    (View,          (S*S, S),           (torch.rand(S, S, S),)                      ),
    (Expand,        (S, 5, S, 5),       ((S, 1, S, 1),)                             ),
    (Exp,           (),                 (torch.rand(S, S, S),)                      ),
    (Log,           (),                 (torch.rand(S, S, S) + 1e-2,)               ),
    (Log1p,         (),                 (torch.rand(S, S, S),)                      ),
    (Tanh,          (),                 ((S, S, S),)                                ),
    (Sigmoid,       (),                 ((S, S, S),)                                ),
    (Sinh,          (),                 ((S, S, S),)                                ),
    (Cosh,          (),                 ((S, S, S),)                                ),
    (Abs,           (),                 ((S, S, S),)                                ),
    (Clamp,         (0, 1),             ((S, S, S),)                                ),
    (Sqrt,          (),                 (torch.rand(S, S, S) + 1e-4,)               ),
    (Sin,           (),                 ((S, S, S),)                                ),
    (Cos,           (),                 ((S, S, S),)                                ),
    (Tan,           (),                 (torch.randn(S, S, S).clamp(-1, 1),)        ),
    (Asin,          (),                 (torch.randn(S, S, S).clamp(-0.9, 0.9),)    ),
    (Acos,          (),                 (torch.randn(S, S, S).clamp(-0.9, 0.9),)    ),
    (Atan,          (),                 ((S, S, S),)                                ),
    (Cinv,          (),                 (torch.rand(S, S, S) + 0.1,)                ),
    (Cmax,          (),                 ((S, S, S), (S, S, S))                      ),
    (Cmin,          (),                 ((S, S, S), (S, S, S))                      ),
    (Round,         (),                 ((S, S, S),)                                ),
    (Sign,          (),                 ((S, S, S),)                                ),
    (Trunc,         (),                 ((S, S, S),)                                ),
    (Floor,         (),                 ((S, S, S),)                                ),
    (Ceil,          (),                 ((S, S, S),)                                ),
    (Frac,          (),                 ((S, S, S),)                                ),
    (Fmod,          (1.5,),             ((S, S, S),)                                ),
    (Lerp,          (0.2,),             ((S, S, S), (S, S, S))                      ),
    (Rsqrt,         (),                 (torch.rand(S, S, S) + 1e-2,)               ),
    (Remainder,     (1.5,),             ((S, S, S),)                                ),
    (CmaxConstant,  (0.5,),             ((S, S, S),)                                ),
    (CminConstant,  (0.5,),             ((S, S, S),)                                ),
    (Mean,          (),                 ((S, S, S),)                                ),
    (Mean,          (1,),               ((S, S, S),),               'dim'           ),
    (Sum,           (),                 ((S, S, S),)                                ),
    (Sum,           (1,),               ((S, S, S),),               'dim'           ),
    (Prod,          (),                 ((S, S, S),)                                ),
    (Prod,          (1,),               ((S, S, S),),               'dim'           ),
    (Addmm,         (),                 ((S, M), (S, S), (S, M)),                   ),
    (Addmm,         (0.1, 1),           ((S, M), (S, S), (S, M)),   'coef'          ),
    (Addbmm,        (),                 ((S, M), (S, S, S), (S, S, M)),             ),
    (Addbmm,        (0.1, 0.4),         ((S, M), (S, S, S), (S, S, M)), 'coef'      ),
    (Baddbmm,       (),                 ((S, S, M), (S, S, S), (S, S, M)),          ),
    (Baddbmm,       (0.1, 0.4),         ((S, S, M), (S, S, S), (S, S, M)), 'coef'   ),
    (Addmv,         (),                 ((S,), (S, M), (M,)),                       ),
    (Addmv,         (0.1, 0.4),         ((S,), (S, M), (M,)),       'coef'          ),
    (Addr,          (),                 ((S, M), (S,), (M,)),                       ),
    (Addr,          (0.1, 0.4),         ((S, M), (S,), (M,)),       'coef'          ),
    (Dot,           (),                 ((L,), (L,)),                               ),
    (Max,           (),                 ((S, S, S),),                               ),
    (Min,           (),                 ((S, S, S),),                               ),
    (Max,           (0,),               ((S, S, S),),               'dim'           ),
    (Min,           (0,),               ((S, S, S),),               'dim'           ),
    (Mode,          (0,),               ((S, S, S),),                               ),
    (Kthvalue,      (2, 0),             ((S, S, S),),                               ),
    (Median,        (0,),               ((S, S, S),),                               ),
    (Norm,          (1.5,),             (torch.rand(S, S, S),),     '1.5'           ),
    (Norm,          (),                 ((S, S, S),),               '2'             ),
    (Norm,          (3,),               ((S, S, S),),               '3'             ),
    (Norm,          (1.5, 0),           (torch.rand(S, S, S),),     '1.5_dim'       ),
    (Norm,          (2, 0),             ((S, S, S),),               '2_dim'         ),
    (Norm,          (3, 0),             ((S, S, S),),               '3_dim'         ),
    (Addcmul,       (),                 ((S, S), (S, S), (S, S))                    ),
    (Addcmul,       (0.6,),             ((S, S), (S, S), (S, S)),   'scale'         ),
    (Addcdiv,       (),                 ((S, S), (S, S), torch.rand(S, S) + 1e-2)   ),
    (Addcdiv,       (0.6,),             ((S, S), (S, S), torch.rand(S, S) + 1e-2), 'scale'),
    (IndexAdd,      (0,),               ((S, S), index_variable(2, S), (2, S))      ),
    (IndexCopy,     (0,),               ((S, S), index_variable(2, S), (2, S))      ),
    (IndexFill,     (0, 2),             ((S, S), index_variable(2, S))              ),
    (IndexSelect,   (0,),               ((S, S), index_variable(2, S))              ),
    (Concat,        (0,),               ((1, S, S), (2, S, S), (3, S, S))           ),
    (Resize,        (S*S, S),           ((S, S, S),)                                ),
    (Diag,          (),                 ((S, S),),                  '2d'            ),
    (Diag,          (),                 ((S,),),                    '1d'            ),
    (Tril,          (),                 ((S, S),)                                   ),
    (Tril,          (2,),               ((S, S),),                  'idx'           ),
    (Triu,          (),                 ((S, S),)                                   ),
    (Triu,          (2,),               ((S, S),),                  'idx'           ),
    (Clone,         (),                 ((S, M, S),)                                ),
    (Squeeze,       (),                 ((S, 1, M, 1),)                             ),
    (Squeeze,       (1,),               ((S, 1, M, 1),),            'dim'           ),
    (Unsqueeze,     (0,),               ((S, M, S),),               '0'             ),
    (Unsqueeze,     (1,),               ((S, M, S),),               '1'             ),
    # (MaskedCopy,    (),                 ((S, S), Variable(torch.randn(S, S).gt(0), requires_grad=False), (S, S),)),
    (MaskedFill,    (10,),              ((S, S), Variable(torch.randn(S, S).gt(0), requires_grad=False))),
    (MaskedSelect,  (),                 ((S, S), Variable(torch.randn(S, S).gt(0), requires_grad=False))),
    (Sort,          (),                 ((S, M, S),)                               ),
    (Sort,          (1,),               ((S, M, S),),               'dim'           ),
    (Sort,          (1, True),          ((S, M, S),),               'dim_desc'      ),
    (Topk,          (3,),               ((S, M, S),)                               ),
    (Topk,          (3, 1),             ((S, M, S),),               'dim'           ),
    (Topk,          (3, 1, True),       ((S, M, S),),               'dim_desc'      ),
    (Topk,          (3, 1, True, True), ((S, M, S),),               'dim_desc_sort' ),
]


method_tests = [
    ('add',         (S, S, S),          ((S, S, S),)                                ),
    ('add',         (S, S, S),          (3.14,),                    'constant'      ),
    ('sub',         (S, S, S),          ((S, S, S),)                                ),
    ('sub',         (S, S, S),          (3.14,),                    'constant'      ),
    ('mul',         (S, S, S),          ((S, S, S),)                                ),
    ('mul',         (S, S, S),          (3.14,),                    'constant'      ),
    ('div',         (S, S, S),          ((S, S, S),)                                ),
    ('div',         (S, S, S),          (3.14,),                    'constant'      ),
    ('pow',         (S, S, S),          ((S, S, S),)                                ),
    ('pow',         (S, S, S),          (3.14,),                    'constant'      ),
    ('transpose',   (1, 2, 3),          (1, 2)                                      ),
    ('t',           (1, 2),             ()                                          ),
    ('view',        (S, S, S),          (S*S, S),                                   ),
    ('view_as',      (S, S, S),          ((S*S, S),)                                 ),
    ('expand',      (S, 1, S),          (S, S, S)                                   ),
    ('exp',         (S, S, S),          ()                                          ),
    ('log',         (S, S, S),          ()                                          ),
    ('log1p',       (S, S, S),          ()                                          ),
    ('tanh',        (S, S, S),          ()                                          ),
    ('sigmoid',     (S, S, S),          ()                                          ),
    ('sinh',        (S, S, S),          ()                                          ),
    ('cosh',        (S, S, S),          ()                                          ),
    ('abs',         (S, S, S),          ()                                          ),
    ('clamp',       (S, S, S),          (0, 1)                                      ),
    ('sqrt',        (S, S, S),          ()                                          ),
    ('sin',         (S, S, S),          ()                                          ),
    ('cos',         (S, S, S),          ()                                          ),
    ('tan',         (S, S, S),          ()                                          ),
    ('asin',        (S, S, S),          ()                                          ),
    ('acos',        (S, S, S),          ()                                          ),
    ('atan',        (S, S, S),          ()                                          ),
    ('cinv',        (S, S, S),          ()                                          ),
    ('round',       (S, S, S),          ()                                          ),
    ('sign',        (S, S, S),          ()                                          ),
    ('trunc',       (S, S, S),          ()                                          ),
    ('floor',       (S, S, S),          ()                                          ),
    ('ceil',        (S, S, S),          ()                                          ),
    ('rsqrt',       (S, S, S),          ()                                          ),
    ('fmod',        (S, S, S),          (1.5,)                                      ),
    ('remainder',   (S, S, S),          (1.5,)                                      ),
    ('lerp',        (S, S, S),          ((S, S, S), 0.4)                            ),
    ('cmax',        (S, S, S),          ((S, S, S),)                                ),
    ('cmax',        (S, S, S),          (0.5,),                     'constant'      ),
    ('cmin',        (S, S, S),          ((S, S, S),)                                ),
    ('cmin',        (S, S, S),          (0.5,),                     'constant'      ),
    ('mean',        (S, S, S),          ()                                          ),
    ('mean',        (S, S, S),          (1,),                       'dim'           ),
    ('sum',         (S, S, S),          ()                                          ),
    ('sum',         (S, S, S),          (1,),                       'dim'           ),
    ('prod',        (S, S, S),          ()                                          ),
    ('prod',        (S, S, S),          (1,),                       'dim'           ),
    ('addmm',       (S, M),             ((S, S), (S, M)),                           ),
    ('addmm',       (S, M),             (0.2, 0.6, (S, S), (S, M)), 'coef'          ),
    ('addbmm',      (S, M),             ((S, S, S), (S, S, M)),                     ),
    ('addbmm',      (S, M),             (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'    ),
    ('baddbmm',     (S, S, M),          ((S, S, S), (S, S, M)),                     ),
    ('baddbmm',     (S, S, M),          (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'    ),
    ('addmv',       (S,),               ((S, M), (M,)),                             ),
    ('addmv',       (S,),               (0.2, 0.6, (S, M), (M,)),   'coef'          ),
    ('addr',        (S, M),             ((S,), (M,)),                               ),
    ('addr',        (S, M),             (0.2, 0.6, (S,), (M,)),     'coef'          ),
    ('dot',         (L,),               ((L,),),                                    ),
    ('max',         (S, S, S),          ()                                          ),
    ('min',         (S, S, S),          ()                                          ),
    ('addcmul',     (S, S),             ((S, S), (S, S))                            ),
    ('addcmul',     (S, S),             (0.5, (S, S), (S, S)),      'scale'         ),
    ('addcdiv',     (S, S),             ((S, S), (S, S))                            ),
    ('addcdiv',     (S, S),             (0.5, (S, S), (S, S)),      'scale'         ),
    ('norm',        (S, S, S),          (2,)                                        ),
    ('norm',        (S, S, S),          (2, 1),                     'dim'           ),
    ('dist',        (S, S, S),          ((S, S, S),)                                ),
    ('dist',        (S, S, S),          ((S, S, S), 4),             '4'             ),
    ('index_select', (S, S, S),         (0, index_variable(2, S))                   ),
    ('diag',        (M, M),             (),                         '2d'            ),
    ('diag',        (M,),               (),                         '1d'            ),
    ('tril',        (M, M),             ()                                          ),
    ('triu',        (M, M),             ()                                          ),
    ('clone',       (S, M, S),          ()                                          ),
    ('permute',     (1, 2, 3, 4),       (0, 2, 3, 1)                                ),
    ('select',      (S, S, S),          (1, 2)                                      ),
    ('narrow',      (S, S, S),          (1, 2, 2)                                   ),
    ('squeeze',     (S, 1, S, 1),       ()                                          ),
    ('squeeze',     (S, 1, S, 1),       (1,),                       '1_dim'         ),
    ('squeeze',     (S, 1, S, 1),       (2,),                       'not_1_dim'     ),
    ('unsqueeze',   (S, S, S),          (0,),                       'first'         ),
    ('unsqueeze',   (S, S, S),          (1,),                       'middle'        ),
    ('unsqueeze',   (S, S, S),          (3,),                       'last'          ),
    ('masked_select', (M, M),           (Variable(torch.ByteTensor(M, M).bernoulli_(), requires_grad=False),)           ),
    ('masked_fill_',  (M, M),           (Variable(torch.ByteTensor(M, M).bernoulli_(), requires_grad=False), 10)        ),
    ('masked_copy_',  (M, M),           (Variable(torch.ByteTensor(M, M).bernoulli_(), requires_grad=False), (M, M))    ),
]
# TODO: mm, bmm, mv, ger
# TODO: max, min with dim (problem with indices)
# TODO: mode, median, sort, kthvalue, topk (problem with indices)
# TODO: indexAdd, indexCopy, indexFill
# TODO: resize, resize_as (tensors only have resize_ and resize_as_)


def create_input(call_args):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)
    def map_arg(arg):
        if isinstance(arg, tuple) and not isinstance(arg[0], Variable):
            return Variable(torch.randn(*arg).double(), requires_grad=True)
        elif torch.is_tensor(arg):
            if isinstance(arg, torch.FloatTensor):
                return Variable(arg.double(), requires_grad=True)
            else:
                return Variable(arg, requires_grad=True)
        else:
            return arg
    return tuple(map_arg(arg) for arg in call_args)


def unpack_variables(args):
    if isinstance(args, Variable):
        return args.data
    elif isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


ignore_inplace = set((
    'test_DivConstant_by_tensor',
))


for test in function_tests:
    cls, constructor_args, call_args = test[:3]
    test_name = 'test_' + cls.__name__ + ('_' + test[3] if len(test) == 4 else '')
    def do_test(self, cls=cls, constructor_args=constructor_args,
            call_args=call_args, test_name=test_name):
        input = create_input(call_args)
        output = cls(*constructor_args)(*input)
        if not isinstance(output, tuple):
            output = (output,)
        for i, o in enumerate(output):
            analytical = get_analytical_jacobian(input, o)
            def fn(input):
                tmp = cls(*constructor_args)(*input)
                if not isinstance(tmp, tuple):
                    tmp = (tmp,)
                return tmp[i].data
            numerical = get_numerical_jacobian(fn, input, input)
            self.assertLessEqual(
                max(a.add(-1, n).abs().max() for a, n in zip(analytical, numerical)),
                PRECISION
            )

        if test_name not in ignore_inplace and issubclass(cls, InplaceFunction):
            inplace_input = deepcopy(input)
            inplace_input_copy = tuple(i + 0 for i in inplace_input)
            fn = cls(*constructor_args, inplace=True)
            inplace_output = fn(*inplace_input_copy)
            if not isinstance(inplace_output, tuple):
                inplace_output = (inplace_output,)
            self.assertEqual(inplace_output, output)
            # Check that gradient is the same
            for inp_i, i in zip(inplace_input, input):
                if inp_i.grad is not None:
                    inp_i.grad.zero_()
                if i.grad is not None:
                    i.grad.zero_()
            for io, o in zip(inplace_output, output):
                grad = torch.randn(*io.size()).double()
                io.backward(grad)
                o.backward(grad)
            for inp_i, i in zip(inplace_input, input):
                self.assertEqual(inp_i.grad, i.grad)

    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


for test in method_tests:
    name, self_size, args = test[:3]
    test_name = 'test_' + name + ('_' + test[3] if len(test) == 4 else '')
    def do_test(self, name=name, self_size=self_size, args=args, test_name=test_name):
        def check(name):
            self_variable = create_input((self_size,))[0]
            args_variable = create_input(args)
            self_tensor = deepcopy(self_variable.data)
            args_tensor = deepcopy(unpack_variables(args_variable))
            output_variable = getattr(self_variable, name)(*args_variable)
            output_tensor = getattr(self_tensor, name)(*args_tensor)
            if not torch.is_tensor(output_tensor) and not isinstance(output_tensor, tuple):
                output_tensor = torch.DoubleTensor((output_tensor,))
            self.assertEqual(unpack_variables(output_variable), output_tensor)
            # TODO: check that both have changed after adding all inplace ops

        check(name)
        inplace_name = name + '_'
        if hasattr(Variable(torch.ones(1)), inplace_name):
            try:
                check(inplace_name)
            except Exception as e:
                if not 'only supports scalar' in e.args[0]:
                    raise


    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


if __name__ == '__main__':
    unittest.main()
