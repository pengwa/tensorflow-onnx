# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for while loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class LoopTests(Tf2OnnxBackendTestBase):

    '''
    def test_simple_while_loop(self):
        #i = tf.constant(0)
        i = tf.placeholder(tf.int32, (), name="input_1")
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

        _ = tf.identity(r, name="output")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32)}

        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)


    def test_simple_while_loop_2(self):
        #i = tf.constant(0)
        i = tf.placeholder(tf.int32, (), name="input_1")
        c = lambda i: tf.logical_and(tf.less(i, 10), tf.greater_equal(i, 3))
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

        _ = tf.identity(r, name="output")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32)}

        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)
    '''

    def test_simple_while_loop_with_ta_write(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        output_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        c = lambda i, *_: tf.logical_and(tf.less(i, 10), tf.greater_equal(i, 3))
        def b(i, out_ta):
            new_i = tf.add(i, 1)
            out_ta_new = out_ta.write(i, i)
            return new_i, out_ta_new

        i_final, ta_final = tf.while_loop(c, b, [i, output_ta])
        r = ta_final.stack()
        _ = tf.identity(r, name="output")
        _ = tf.identity(i_final, name="i")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32)}

        output_names_with_port = ["output:0", "i:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

if __name__ == '__main__':
    Tf2OnnxBackendTestBase.trigger(LoopTests)
