# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.loop_rewriter - generic loop support
"""

from __future__ import division
from __future__ import print_function
import logging
import sys
import traceback
from onnx import onnx_pb, TensorProto
import numpy as np
from tf2onnx.graph import Graph
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase, Context
from tf2onnx.rewriter.rnn_utils import is_tensor_array_gather_op, is_tensor_array_write_op
from tf2onnx.rewriter.rnn_utils import BodyGraphDict, REWRITER_RESULT, SubGraphMetadata
from tf2onnx.tfonnx import utils


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.loop_rewriter")


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,broad-except,protected-access


class LoopRewriter(LoopRewriterBase):
    def __init__(self, g):
        super(LoopRewriter, self).__init__(g)

    def create_context(self):
        return Context()

    def run(self):
        log.debug("enter loop rewriter")
        return self.run_internal()

    def need_rewrite(self, context):
        return True

    def rewrite(self, context):
        log.debug("enter rewrite function")
        loop_node = None
        try:
            loop_props = context.loop_properties
            cell_g_info = context.cell_graph
            cond_g_info = context.cond_graph

            # first_time_cond_g_info = copy.deepcopy(cond_g_info)

            # replace condition graph's inputs to be cell graph's outputs, because we want condition graph
            # to consumer cell graph outputs.
            for loop_var in cond_g_info.dependent_vars:
                self.g.replace_all_inputs(cond_g_info.nodes, loop_var.switch_true_identity_output_id,
                                          loop_var.next_iteration_input_id)

            body_nodes = set(cell_g_info.nodes + cond_g_info.nodes)
            body_outputs = cond_g_info.outputs + cell_g_info.outputs
            loop_body_g = LoopRewriterBase.construct_graph_from_nodes(self.g, body_nodes, body_outputs)

            # create loop body graph inputs
            iter_name = utils.make_name("i")
            cond_name = utils.make_name("cond")
            iter_num_input = utils.make_onnx_inputs_outputs(iter_name, TensorProto.INT64, ())
            cond_input = utils.make_onnx_inputs_outputs(cond_name, TensorProto.BOOL, ())
            loop_body_g_inputs = [iter_num_input, cond_input]

            for i, input_name in enumerate(loop_props.state_inputs):
                if input_name is not None:
                    dtype = self.g.get_dtype(input_name)
                    shape = self.g.get_shape(input_name)
                else:
                    # if the variable is not used in the body graph, then we created a fake one,
                    # the same type and shape as its corresponding output.
                    output_id = loop_props.state_outputs[i]
                    dtype = self.g.get_dtype(output_id)
                    shape = self.g.get_shape(output_id)
                    input_name = utils.make_name("unused_state_input_")

                val = utils.make_onnx_inputs_outputs(input_name, dtype, shape)
                loop_body_g_inputs.append(val)

            body_nodes_to_append = []
            for input_ta in context.input_tas:
                # Loop does not have scan inputs, so we use Gather to get data for each iteration.
                unsqueezed_index_node = loop_body_g.make_node("Unsqueeze", [input_ta.index_input_id], attr={"axes": [0]})
                body_nodes_to_append.append(unsqueezed_index_node)
                gather_node = loop_body_g.make_node("Gather", [input_ta.data_input_id, unsqueezed_index_node.output[0]])
                body_nodes_to_append.append(gather_node)
                data_node = loop_body_g.make_node("Squeeze", [gather_node.output[0]], attr={"axes": [0]})
                body_nodes_to_append.append(data_node)

                loop_body_g.replace_all_inputs(loop_body_g.get_nodes(), input_ta.output_id, data_node.output[0])

            for i in loop_body_g_inputs:
                loop_body_g.add_model_input(i)

            body_nodes = loop_body_g.get_nodes()
            body_nodes.extend(body_nodes_to_append)
            loop_body_g.set_nodes(body_nodes)

            loop_node = self._create_loop_node(context, loop_props)
            if not loop_node:
                log.error("failed to create loop node during rewrite")
                return REWRITER_RESULT.FAIL

            loop_node.set_body_graph_as_attr("body", loop_body_g)

            all_nodes = self.g.get_nodes()
            all_nodes.append(loop_node)
            self.g.set_nodes(all_nodes)
            return REWRITER_RESULT.OK

        except Exception as ex:
            tb = traceback.format_exc()
            log.error("rewrite failed, due to exception: %s, details:%s", ex, tb)
            return REWRITER_RESULT.FAIL

    def _create_loop_node(self, context, loop_props):
        log.debug("create loop node")

        # reuse original output connection id (e.g. Exit_XXX), so we don't need set shape.
        loop_outputs = []
        for exit_output_id in context.loop_properties.state_output_exit_ids + context.loop_properties.scan_output_exit_ids:
            if exit_output_id:
                loop_outputs.append(exit_output_id)
            else:
                loop_outputs.append(utils.make_name("used_loop_output_"))

        # trip count and cond are not used, giving them values just because bug
        # (https://github.com/Microsoft/onnxruntime/issues/255) of onnxruntime.
        trip_cnt = self.g.make_const(utils.make_name("trip_count"), np.array(sys.maxsize, dtype=np.int64))
        cond = self.g.make_const(utils.make_name("cond"), np.array(True, dtype=np.bool))
        loop_node = self.g.make_node("Loop", [trip_cnt.output[0]] + [cond.output[0]] +
                                     loop_props.initial_state_variable_values,
                                     outputs=loop_outputs,
                                     skip_conversion=False)

        return loop_node
