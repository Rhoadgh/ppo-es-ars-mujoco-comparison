import numpy as np
import tensorflow as tf
import builtins
import functools
import copy
import os

# Set up the Compatibility Bridge
tf1 = tf.compat.v1
tf1.disable_v2_behavior()

# Try to start a session if one doesn't exist
try:
    tf1.InteractiveSession()
except:
    pass

# ================================================================
# Import all names into common namespace
# ================================================================

clip = tf1.clip_by_value

# Make consistent with numpy
# ----------------------------------------

def sum(x, axis=None, keepdims=False):
    # Note: reduction_indices is old, axis is modern
    return tf1.reduce_sum(x, axis=axis, keep_dims=keepdims)

def mean(x, axis=None, keepdims=False):
    return tf1.reduce_mean(x, axis=axis, keep_dims=keepdims)

def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf1.square(x - meanx), axis=axis, keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return tf1.sqrt(var(x, axis=axis, keepdims=keepdims))

def max(x, axis=None, keepdims=False):
    return tf1.reduce_max(x, axis=axis, keep_dims=keepdims)

def min(x, axis=None, keepdims=False):
    return tf1.reduce_min(x, axis=axis, keep_dims=keepdims)

def concatenate(arrs, axis=0):
    return tf1.concat(arrs, axis)

def argmax(x, axis=None):
    return tf1.argmax(x, axis=axis)

def switch(condition, then_expression, else_expression):
    x_shape = copy.copy(then_expression.get_shape())
    x = tf1.cond(tf1.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

# Extras
# ----------------------------------------
def l2loss(params):
    if len(params) == 0:
        return tf1.constant(0.0)
    else:
        return tf1.add_n([sum(tf1.square(p)) for p in params])

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * builtins.abs(x)

def categorical_sample_logits(X):
    U = tf1.random_uniform(tf1.shape(X))
    return argmax(X - tf1.log(-tf1.log(U)), axis=1)

# ================================================================
# Global session
# ================================================================

def get_session():
    return tf1.get_default_session()

def single_threaded_session():
    tf_config = tf1.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    return tf1.Session(config=tf_config)

ALREADY_INITIALIZED = set()
def initialize():
    # Update to modern variable initialization
    new_variables = set(tf1.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf1.variables_initializer(list(new_variables)))
    ALREADY_INITIALIZED.update(new_variables)

def eval(expr, feed_dict=None):
    if feed_dict is None: feed_dict = {}
    return get_session().run(expr, feed_dict=feed_dict)

def set_value(v, val):
    get_session().run(v.assign(val))

def load_state(fname):
    saver = tf1.train.Saver()
    saver.restore(get_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf1.train.Saver()
    saver.save(get_session(), fname)

# ================================================================
# Model components
# ================================================================

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf1.constant(out)
    return _initializer

def dense(x, size, name, weight_init=None, bias=True):
    with tf1.variable_scope(name):
        w = tf1.get_variable("w", [x.get_shape()[1], size], initializer=weight_init)
        ret = tf1.matmul(x, w)
        if bias:
            b = tf1.get_variable("b", [size], initializer=tf1.zeros_initializer())
            return ret + b
        else:
            return ret

# ================================================================
# Basic Stuff
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, dict):
        f = _Function(inputs, list(outputs.values()), updates, givens=givens)
        return lambda *inputs : dict(zip(outputs.keys(), f(*inputs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *inputs : f(*inputs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf1.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(zip(self.inputs, inputvals))
        feed_dict.update(self.givens)
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results

# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list):
    grads = tf1.gradients(loss, var_list)
    return tf1.concat([tf1.reshape(grad, [numel(v)])
        for (v, grad) in zip(var_list, grads)], 0)

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf1.float32):
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf1.placeholder(dtype,[total_size])
        start=0
        assigns = []
        for (shape,v) in zip(shapes,var_list):
            size = intprod(shape)
            assigns.append(tf1.assign(v, tf1.reshape(theta[start:start+size],shape)))
            start+=size
        self.op = tf1.group(*assigns)
    def __call__(self, theta):
        get_session().run(self.op, feed_dict={self.theta:theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf1.concat([tf1.reshape(v, [numel(v)]) for v in var_list], 0)
    def __call__(self):
        return get_session().run(self.op)

# ================================================================
# Misc
# ================================================================

def scope_vars(scope, trainable_only):
    return tf1.get_collection(
        tf1.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf1.GraphKeys.VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )

def in_session(f):
    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        with tf1.Session():
            f(*args, **kwargs)
    return newfunc

_PLACEHOLDER_CACHE = {}
def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        return out
    else:
        out = tf1.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out,dtype,shape)
        return out

def flattenallbut0(x):
    return tf1.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])

def reset():
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    tf1.reset_default_graph()