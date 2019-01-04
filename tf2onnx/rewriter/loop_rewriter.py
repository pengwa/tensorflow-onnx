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
            cell_g = LoopRewriterBase.construct_graph_from_nodes(self.g, cell_g_info.nodes, cell_g_info.outputs)
            cond_g = LoopRewriterBase.construct_graph_from_nodes(self.g, cond_g_info.nodes, cond_g_info.outputs)

            # handle cell graph: insert identity node, since sometimes we need output same output_id 
            # as state_output and scan_out, but ONNX don't allow the same output_id appeared more
            # than once as output node.
            cell_body_nodes = []
            cell_g.output_names = []
            for o in cell_g_info.outputs:
                node = cell_g.make_node("Identity", inputs=[o], shapes=[cell_g.get_shape(o)],
                                        dtypes=[cell_g.get_dtype(o)])
                cell_g.output_names.append(node.output[0])
                cell_body_nodes.append(node)

            cell_nodes = cell_g.get_nodes()
            cell_nodes.extend(cell_body_nodes)
            cell_g.set_nodes(cell_nodes)


            # handle else_branch graph: use cell graph inputs as inputs of If's else_branch graph.
            else_g_nodes = []
            else_g_output_names = []
            for c_i in cell_g_info.inputs:
                node = self.g.make_node("Identity", inputs=[c_i], shapes=[self.g.get_shape(c_i)],
                                        dtypes=[self.g.get_dtype(c_i)])
                else_g_nodes.append(node)
                else_g_output_names.append(node.output[0])
            else_g = LoopRewriterBase.construct_graph_from_nodes(self.g, else_g_nodes, else_g_output_names)


            # handle condition graph: we will use this graph as the Loop's body graph.
            nodes_to_g = []
            # should not skip conversion, because its body graph needs to be processed during conversion.
            utils.make_sure(len(cond_g_info.outputs) == 1, "condition graph should have only one output.")
            if_op = cond_g.make_node("If", cond_g_info.outputs, output_count=len(cell_g_info.outputs),
                                     shapes=[cell_g.get_shape(o) for o in cell_g_info.outputs], 
                                     dtypes=[cell_g.get_dtype(o) for o in cell_g_info.outputs],
                                     skip_conversion=False)
            if_op.set_body_graph_as_attr("then_branch", cell_g)
            if_op.set_body_graph_as_attr("else_branch", else_g)

            # must update proto to let 
            if_op.update_proto()
            nodes_to_g.append(if_op)

            iter_name = utils.make_name("i")
            cond_name = utils.make_name("cond")
            iter_num_input = utils.make_onnx_inputs_outputs(iter_name, TensorProto.INT64, ())
            cond_input = utils.make_onnx_inputs_outputs(cond_name, TensorProto.BOOL, ())
            cond_g.add_model_input(iter_num_input)
            cond_g.add_model_input(cond_input)

            for input_name, init_input_id in zip(loop_props.loop_state_inputs, loop_props.initial_state_inputs):
                shape = self.g.get_shape(input_name)
                dtype = self.g.get_dtype(input_name)
                if shape is None:
                    shape = self.g.get_shape(init_input_id)
                    loop_input_shape = list(shape)
                else:
                    loop_input_shape = list(shape)

                val = utils.make_onnx_inputs_outputs(input_name, dtype, loop_input_shape)
                cond_g.add_model_input(val)

            for input_name, init_input_id in zip(loop_props.loop_scan_inputs, loop_props.initial_scan_inputs):
                shape = self.g.get_shape(input_name)
                dtype = self.g.get_dtype(input_name)
                if shape is None:
                    shape = self.g.get_shape(init_input_id)
                    loop_input_shape = list(shape)
                else:
                    loop_input_shape = list(shape)

                loop_input_shape = list(shape)[1:]
                val = utils.make_onnx_inputs_outputs(input_name, dtype, loop_input_shape)
                cond_g.add_model_input(val)

            log.debug("start preparing body graph outputs nodes")

            # create "cond" output of the Loop's body graph
            cond_output_id = cond_g_info.outputs[0]
            cond_identity = cond_g.make_node("Identity", inputs=cond_g_info.outputs, shapes=[cond_g.get_shape(cond_output_id)],
                                             dtypes=[cond_g.get_dtype(cond_output_id)])
            nodes_to_g.append(cond_identity)

            cond_g_nodes = cond_g.get_nodes()
            cond_g_nodes.extend(nodes_to_g)
            cond_g.set_nodes(cond_g_nodes)
            cond_g.output_names = [cond_identity.output[0]] + if_op.output

            loop_node = self._create_loop_node(context, loop_props)
            if not loop_node:
                log.error("failed to create loop node during rewrite")
                return REWRITER_RESULT.FAIL

            loop_node.set_body_graph_as_attr("body", cond_g)
            loop_node.update_proto()
            nodes_to_append = []
            nodes_to_append.append(loop_node)

            log.debug("add body graph meta data into store")

            self._connect_loop_with_output(context, loop_node)

            all_nodes = self.g.get_nodes()
            all_nodes.extend(nodes_to_append)
            self.g.set_nodes(all_nodes)

            return REWRITER_RESULT.OK
        except Exception as ex:
            tb = traceback.format_exc()
            log.error("rewrite failed, due to exception: %s, details:%s", ex, tb)
            return REWRITER_RESULT.FAIL

    def _create_loop_node(self, context, loop_props):
        log.debug("create loop node")
        # trip count and cond are not used, giving them values just because bug
        # (https://github.com/Microsoft/onnxruntime/issues/255) of onnxruntime.
        trip_cnt = self.g.make_const(utils.make_name("trip_count"), np.array(sys.maxsize, dtype=np.int64))
        cond = self.g.make_const(utils.make_name("cond"), np.array(True, dtype=np.bool))
        loop_node = self.g.make_node("Loop", [trip_cnt.output[0]] + [cond.output[0]] + 
                                     loop_props.initial_state_inputs + loop_props.initial_scan_inputs,
                                     output_count=len(loop_props.loop_state_outputs + loop_props.loop_scan_outputs),
                                     skip_conversion=False)

        # todo: set output shape for the loop
        return loop_node

    def _connect_loop_with_output(self, context, loop_node):
        log.debug("connect scan output with the graph")
        index = 0
        for _, val in context.loop_variables.items():
            var_output_id = val.exit_output_id
            if var_output_id:
                self.g.replace_all_inputs(self.g.get_nodes(), var_output_id, loop_node.output[index])

            index += 1
