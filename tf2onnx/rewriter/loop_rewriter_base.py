# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.loop_rewriter_base
"""

from __future__ import division
from __future__ import print_function
import copy
import logging
from collections import deque
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.rewriter.rnn_utils import is_loopcond_op, is_tensor_array_op, is_tensor_array_write_op
from tf2onnx.rewriter.rnn_utils import is_tensor_array_gather_op, is_tensor_array_write_op
from tf2onnx.rewriter.rnn_utils import BodyGraphDict, REWRITER_RESULT, SubGraphMetadata
from tf2onnx.graph import Node, Graph

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.loop_rewriter_base")
INVLAID_INPUT_ID = "invalid:0"
# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class Context(object):
    def __init__(self):
        self.need_keep_nodes = []
        self.while_context_scope = None
        self.loop_variables = {}
        self.loop_cond = None

        self.input_tas = []
        self.loop_properties = None
        self.cell_graph = None
        self.cond_graph = None

class GraphInfo(object):
    def __init__(self, ops, inputs, outputs):
        self.nodes = ops
        self.inputs = inputs
        self.outputs = outputs

class LoopVariable(object):
    def __init__(self, enter_name, enter_input_id, next_iteration_input_id,
                 switch_true_identity_output_id, exit_output_id, is_tensor_array):
        self.enter_name = enter_name
        self.enter_input_id = enter_input_id
        self.next_iteration_input_id = next_iteration_input_id
        self.switch_true_identity_output_id = switch_true_identity_output_id
        self.exit_output_id = exit_output_id

        self.is_tensor_array = is_tensor_array
        self.ta_index_id = None


class LoopProperties(object):
    def __init__(self, initial_state_inputs, initial_scan_inputs, loop_state_inputs,
                 loop_state_outputs, loop_scan_inputs, loop_scan_outputs):
        self.initial_state_inputs = initial_state_inputs
        self.initial_scan_inputs = initial_scan_inputs
        self.loop_state_inputs = loop_state_inputs
        self.loop_state_outputs = loop_state_outputs
        self.loop_scan_inputs = loop_scan_inputs
        self.loop_scan_outputs = loop_scan_outputs


class TensorArrayProp(object):
    def __init__(self):
        self.index_input_id = None
        self.data_input_id = None
        self.output_id = None


class LoopRewriterBase(object):
    def __init__(self, g):
        self.g = g
        self.ta_read_input_pattern = \
            OpTypePattern('TensorArrayReadV3', name='ta_read', inputs=[
                OpTypePattern("Enter", name="ta_enter", inputs=[
                    OpTypePattern("TensorArrayV3")
                ]),
                OpTypePattern('*'),
                OpTypePattern("Enter", name="ta_scatter_enter", inputs=[
                    OpTypePattern("TensorArrayScatterV3", name="ta_input_scatter")
                ]),
            ])

    def create_context(self):
        return Context()

    def need_rewrite(self, context):
        return False

    def rewrite(self, context):
        return REWRITER_RESULT.FAIL

    def run_internal(self):
        log.debug("enter loop rewriter")
        for n in self.g.get_nodes():
            if is_loopcond_op(n):
                log.debug("======================")
                log.debug("found LoopCond op named %s", n.name)
                context = self.create_context()
                context.loop_cond = n
                self._parse_loop_variables(n, context)
                self._parse_input_ta(context)
                if self.need_rewrite(context):
                    self._compose_body_graph_inputs_and_outputs(context)
                    # cut off connection between cell/cond graphs and useless node like Merge, NextIteration.
                    to_remove = self._cut_off_connection_for_cell(context)
                    all_nodes = self.g.get_nodes()
                    for n in set(to_remove):
                        if n in all_nodes:
                            all_nodes.remove(n)
                    self.g.set_nodes(all_nodes)

                    context.cell_graph = self._crop_loop_body_sub_graph(context)
                    context.cond_graph = self._crop_loop_condition_sub_graph(context)

                    _result = self.rewrite(context)
                    if _result == REWRITER_RESULT.OK:
                        log.debug("rewrite successfully")
                    elif _result == REWRITER_RESULT.SKIP:
                        log.debug("rewrite skipped for LoopCond called %s", n.name)
                        continue
                    elif _result == REWRITER_RESULT.FAIL:
                        raise ValueError("rewrite failed, so just fast fail it")
        all_output_name = copy.deepcopy(self.g.output_names)

        if all_output_name:
            #all_output_name.extend(BodyGraphDict.get_body_graph_output_names())
            self.g.delete_unused_nodes(all_output_name)

        return self.g.get_nodes()

    def _parse_loop_variables(self, loop_cond_op, context):
        parts = loop_cond_op.name.split('/')
        context.while_context_scope = '/'.join(parts[0:-1]) + "/"
        log.debug("found while loop scope %s", context.while_context_scope)

        switch_nodes = self.g.find_output_consumers(loop_cond_op.output[0])
        for s in switch_nodes:
            if s.type != 'Switch':
                raise ValueError("LoopCond's output node should be followed with a Switch node")

            loop_var = self._get_loop_var_from_switch(s)
            if loop_var.enter_name in context.loop_variables:
                raise ValueError("duplicated enter name registered")

            context.loop_variables[loop_var.enter_name] = loop_var

    def _parse_output_ta(self, loop_var):
        # here we parse patterns generated by 
        # ta.write(), then ta.stack(), because this is the most frequent usage pattern.
        ta_write_node = self.g.get_node_by_output(loop_var.next_iteration_input_id)
        utils.make_sure(is_tensor_array_write_op(ta_write_node), "ta var nextiteration is not following ta write op")
        loop_var.next_iteration_input_id = ta_write_node.input[2]

        if loop_var.exit_output_id:
            exit_consumers = self.g.find_output_consumers(loop_var.exit_output_id)
            ta_gather_node = [n for n in exit_consumers if is_tensor_array_gather_op(n)][0]

            # update exit output id, treat the gather output as ta's output
            loop_var.exit_output_id = ta_gather_node.output[0]

        log.debug("output ta %s - next_iteration_input (%s) shape: %s, output (%s) shape: %s", loop_var.enter_name,
                    loop_var.next_iteration_input_id, self.g.get_shape(loop_var.next_iteration_input_id),
                    loop_var.exit_output_id, self.g.get_shape(loop_var.exit_output_id))

    def _parse_input_ta(self, context):
        matcher = GraphMatcher(self.ta_read_input_pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(self.g.get_nodes()))
        match_results = [r for r in match_results if r.get_op("ta_input_scatter").name.startswith(context.rnn_scope)]
        for match in match_results:
            ta_input_scatter = match.get_op("ta_input_scatter")
            # the 3rd input of scatter is the value
            input_ta = TensorArrayProp()

            # dynamic_rnn specific approach.
            input_ta.data_input_id = ta_input_scatter.input[2]

            ta_read_node = match.get_op("ta_read")
            input_ta.index_input_id = ta_read_node.input[1]
            input_ta.output_id = match.get_op("ta_read").output[0]

            input_shape = self.g.get_shape(input_ta.data_input_id)
            output_shape = self.g.get_shape(input_ta.output_id)
            if output_shape is None and input_shape is not None:
                self.g.set_shape(input_ta.output_id, input_shape[1:])

            context.input_tas.append(input_ta)

            log.debug("input ta %s - data input (%s) shape: %s, output (%s) shape: %s", ta_read_node.name,
                      input_ta.data_input_id, self.g.get_shape(input_ta.data_input_id),
                      input_ta.output_id, self.g.get_shape(input_ta.output_id))

    def _crop_loop_body_sub_graph(self, context):
        # according to input and output, find the body graph
        loop_props = context.loop_properties
        input_ids = loop_props.loop_state_inputs + loop_props.loop_scan_inputs
        output_ids = loop_props.loop_state_outputs + loop_props.loop_scan_outputs

        ops, enter_nodes, _ = self.find_subgraph(set(input_ids), set(output_ids), self.g, merge_as_end=False)

        other_enter_input_ids = []
        for enter_node in enter_nodes:
            # connect Enter's output to Enter's input
            self.g.replace_all_inputs(ops, enter_node.output[0], enter_node.input[0])
            other_enter_input_ids.append(enter_node.input[0])

        return GraphInfo(ops, input_ids, output_ids)

    def _crop_loop_condition_sub_graph(self, context):
        # according to input and output, find the body graph
        loop_props = context.loop_properties
        input_ids = loop_props.loop_state_inputs + loop_props.loop_scan_inputs
        output_ids = [context.loop_cond.input[0]]
        ops, enter_nodes, merge_nodes = self.find_subgraph(set(input_ids), set(output_ids), self.g, merge_as_end=True)

        other_enter_input_ids = []
        for enter_node in enter_nodes:
            # connect Enter's output to Enter's input
            self.g.replace_all_inputs(ops, enter_node.output[0], enter_node.input[0])
            other_enter_input_ids.append(enter_node.input[0])

        for merge_node in merge_nodes:
            enter_node = [n for n in merge_node.inputs if n.type == "Enter"][0]
            loop_var = context.loop_variables[enter_node.name]

            # cut off connection between condition graph and Merge node.
            non_switch_consumers = [n for n in self.g.find_output_consumers(merge_node.output[0]) if n.type != "Switch"]
            self.g.replace_all_inputs(non_switch_consumers, merge_node.output[0], loop_var.switch_true_identity_output_id)
            utils.make_sure(loop_var.switch_true_identity_output_id in input_ids,
                            "Merge op related loop var should already in the inputs")

        # cut off connection between condition graph and LoopCond node.
        self.g.replace_all_inputs([context.loop_cond], context.loop_cond.output[0], INVLAID_INPUT_ID)

        return GraphInfo(ops, input_ids, output_ids)

    def _cut_off_connection_for_cell(self, context):
        nodes_to_remove = []
        for val in context.loop_variables.values():
            # remove the node to cut off a starting node of the cell (e.g. loop body).
            nodes_to_remove.append(self.g.get_node_by_output(val.switch_true_identity_output_id))

            if val.is_tensor_array:
                # remove the node to cut off connection between scan_output and the cell.
                ta_write_nodes = [n for n in self.g.get_nodes() if is_tensor_array_write_op(n)]
                self.g.replace_all_inputs(ta_write_nodes, val.next_iteration_input_id, INVLAID_INPUT_ID)
            else:
                # connect NextIteration to an invalid node, to cut off a ending node of the cell.
                next_iter_nodes = [n for n in self.g.get_nodes() if n.type == "NextIteration"]
                self.g.replace_all_inputs(next_iter_nodes, val.next_iteration_input_id, INVLAID_INPUT_ID)

        for input_ta in context.input_tas:
            # remove the node to cut off connection between scan_input and the cell.
            nodes_to_remove.append(self.g.get_node_by_output(input_ta.output_id))

        return nodes_to_remove

    def _compose_body_graph_inputs_and_outputs(self, context):
        log.debug("_compose_body_inputs_and_outputs")

        loop_state_inputs = []
        loop_state_outputs = []
        loop_scan_outputs = []
        initial_state_inputs = []
        initial_scan_inputs = []
        # put state variables ahead of scan variables.
        for var in context.loop_variables.values():
            if var.is_tensor_array:
                continue
            loop_state_inputs.append(var.switch_true_identity_output_id)
            loop_state_outputs.append(var.next_iteration_input_id)
            initial_state_inputs.append(var.enter_input_id)

        for var in context.loop_variables.values():
            if not var.is_tensor_array:
                continue
            log.debug("prepare cell scan outputs")
            self._parse_output_ta(var)
            loop_scan_outputs.append(var.next_iteration_input_id)

        log.debug("prepare cell scan inputs")
        loop_scan_inputs = []
        for input_ta in context.input_tas:
            loop_scan_inputs.append(input_ta.output_id)
            initial_scan_inputs.append(input_ta.data_input_id)

        loop_props = LoopProperties(initial_state_inputs, initial_scan_inputs, loop_state_inputs, loop_state_outputs,
                                    loop_scan_inputs, loop_scan_outputs)
        context.loop_properties = loop_props
        return loop_props

    def _get_loop_var_from_switch(self, switch_node):
        if switch_node.type != 'Switch':
            log.error("not a switch node, skip")
            return None

        # the first input is data
        merge_node = switch_node.inputs[0]
        if merge_node.type != "Merge":
            log.error("switch node does not has Merge as its first input")
            return None

        # find the output_true consumers
        switch_consumers = self.g.find_output_consumers(switch_node.output[1])
        if len(switch_consumers) != 1:
            raise ValueError("switch has non-1 consumers")

        if switch_consumers[0].type != "Identity":
            raise ValueError("switch has consumer that is not Identity")
        identity_node = switch_consumers[0]

        target_node_input_id = None
        enter_node = [n for n in merge_node.inputs if n.type == 'Enter'][0]
        target_node_input_id = enter_node.input[0]
        log.debug("a Switch >> Merge >> Enter is found called %s", enter_node.inputs[0].name)

        next_iteration_node = [n for n in merge_node.inputs if n.type == 'NextIteration'][0]
        last_iteration_output_id = next_iteration_node.input[0]

        # find the output_false consumers to see whether there is consumer for this var
        switch_false_consumers = self.g.find_output_consumers(switch_node.output[0])
        false_consumer_count = len(switch_false_consumers)
        exit_output_id = None
        if false_consumer_count == 1:
            exit_node = switch_false_consumers[0]
            if exit_node.type != "Exit":
                raise ValueError("switch false branch is followed by non-Exit")
            exit_output_id = exit_node.output[0]
        elif false_consumer_count == 0:
            exit_output_id = None
        else:
            raise ValueError("unexpected number of switch false consumers")

        is_ta = False
        if is_tensor_array_op(self.g.get_node_by_output(target_node_input_id)):
            is_ta = True

        loop_var = LoopVariable(enter_node.name, target_node_input_id, last_iteration_output_id,
                                identity_node.output[0], exit_output_id, is_ta)
        loop_var = self._tune_shape_for_loop_var(loop_var)
        #loop_var = self._tune_shape_for_loop_ta_var(loop_var)
        return loop_var

    def _tune_shape_for_loop_ta_var(self, loop_var):
        if loop_var.is_tensor_array:
            ta_write_node = self.g.get_node_by_output(loop_var.next_iteration_input_id)
            if not is_tensor_array_write_op(ta_write_node):
                raise ValueError("ta var nextiteration is not following ta write op")

            loop_var.next_iteration_input_id = ta_write_node.input[2]
            loop_var.ta_index_id = ta_write_node.input[1]

            ta_output_shape = None
            next_iteration_shape = self.g.get_shape(loop_var.next_iteration_input_id)
            if next_iteration_shape is None:
                enter_node = ta_write_node.inputs[0]
                ta_node_output = enter_node.input[0]
                ta_element_shape = self.g.get_shape(ta_node_output)
                ta_output_shape = ta_element_shape
                log.debug("loop var [%s, %s] output shapes are inferred from TA element shape", loop_var.enter_name,
                          loop_var.enter_input_id)
            else:
                log.debug("loop var [%s, %s] output shapes are inferred from cell output %s", loop_var.enter_name,
                          loop_var.enter_input_id, loop_var.next_iteration_input_id)
                ta_output_shape = next_iteration_shape

            self.g.set_shape(loop_var.next_iteration_input_id, ta_output_shape)
            self.g.set_shape(loop_var.switch_true_identity_output_id, ta_output_shape)
            self.g.set_shape(loop_var.exit_output_id, ta_output_shape)

        return loop_var

    def _tune_shape_for_loop_var(self, loop_var):
        if loop_var.is_tensor_array:
            return loop_var
        log.debug("_tune_shape_for_loop_var for loop var [%s, %s, %s]", loop_var.enter_name,
                  loop_var.enter_input_id, loop_var.next_iteration_input_id)
        var_output_shape = self.g.get_shape(loop_var.enter_input_id)
        if var_output_shape is None:
            var_output_shape = self.g.get_shape(loop_var.next_iteration_input_id)

        self.g.set_shape(loop_var.next_iteration_input_id, var_output_shape)
        self.g.set_shape(loop_var.switch_true_identity_output_id, var_output_shape)
        self.g.set_shape(loop_var.exit_output_id, var_output_shape)
        log.debug("_tune_shape_for_loop_var new shape is %s", var_output_shape)

        return loop_var

    @staticmethod
    def find_subgraph(input_ids, output_ids, g, merge_as_end=False):
        log.info("input ids %s ", input_ids)
        log.info("output ids %s ", output_ids)

        enter_nodes = set()
        merge_nodes = set()

        def find_input_boundary(node):
            if node.type == "Enter":
                enter_nodes.add(node)
                log.debug("terminate the input search at %s", node.name)
                return False
            elif merge_as_end is True and node.type == "Merge":
                merge_nodes.add(node)
                log.debug("terminate the input search at %s", node.name)
                return False
            elif node.is_const():
                log.debug("terminate search at const node %s", node.name)
                return False

            for o in node.output:
                if o in input_ids:
                    return False
            return True

        nodes = g.extract_sub_graph_nodes(output_ids, input_checker=find_input_boundary)
        return nodes, enter_nodes, merge_nodes

    @staticmethod
    def construct_graph_from_nodes(parent_g, nodes, output_ids):
        log.debug("construct_graph_from_nodes")

        nodes = set(nodes)
        all_inputs_and_outputs = set()
        ops = []
        for op in nodes:
            all_inputs_and_outputs |= set(op.input)
            all_inputs_and_outputs |= set(op.get_implicit_inputs())
            all_inputs_and_outputs |= set(op.output)
            op.update_proto()
            onnx_op = op.op
            ops.append(onnx_op)

        out_shapes = {}
        out_dtypes = {}
        for i in all_inputs_and_outputs:
            if i not in out_shapes:
                out_shapes[i] = parent_g._output_shapes[i]
            if i not in out_dtypes:
                out_dtypes[i] = parent_g._dtypes[i]

        g = Graph(ops, output_shapes=out_shapes, dtypes=out_dtypes, output_names=output_ids)
        return g