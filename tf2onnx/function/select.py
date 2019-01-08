# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - select op conversion
"""
import numpy as np
from onnx import helper, onnx_pb
from onnx.onnx_pb import TensorProto
from tf2onnx import utils
from tf2onnx.graph import Node, Graph
from tf2onnx.utils import port_name, make_sure


# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring


def select_op8(ctx, node, name, args):
    # T output = Select(bool condition, T x, T y)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    utils.make_sure(len(node.input) > 1, "Select with only condition is not supported.")

    nodes = []
    true_data_type = ctx.get_dtype(node.input[1])
    false_data_type = ctx.get_dtype(node.input[2])
    true_data_shape = ctx.get_shape(node.input[1])
    false_data_shape = ctx.get_shape(node.input[2])
    make_sure(true_data_type == false_data_type, "select true val and false val have different data types.")
    make_sure(np.array_equal(true_data_shape, false_data_shape),
              "select true val and false val have different output shapes.")

    condition_shape = ctx.get_shape(node.input[0])
    utils.make_sure(condition_shape is not None, "condition shape is None")
    rank = len(condition_shape)

    utils.make_sure(rank >= 0, "rank should be >= 0")
    val_output_id = None
    if rank > 0:
        # create nodes getting shape of condition
        shape_node_output_shape = [rank]
        shape_node = ctx.make_node("Shape", [node.input[0]], op_name_scope=node.name,
                                   shapes=[shape_node_output_shape], dtypes=[onnx_pb.TensorProto.INT64])
        nodes.append(shape_node)

        # todo(pengwa), move those leveraging rewrite_incomplete_type_support_onnxruntime after shape inferencing
        # bug is fixed.
        # workaround: onnxruntime does not support Split-2, add cases before and after.
        target_dtype = onnx_pb.TensorProto.FLOAT
        shape_f_node = ctx.make_node("Cast", [shape_node.output[0]], attr={"to": target_dtype},
                                     shapes=[shape_node_output_shape], dtypes=[target_dtype],
                                     op_name_scope=node.name)
        nodes.append(shape_f_node)

        split_attr = [1 for i in range(rank)]
        output_shapes = [[1] for i in range(rank)]
        output_dtypes = [target_dtype for i in range(rank)]
        split_node = ctx.make_node("Split", [shape_f_node.output[0]], output_count=rank,
                                   attr={"split": split_attr}, shapes=output_shapes,
                                   dtypes=output_dtypes, op_name_scope=node.name)
        nodes.append(split_node)

        trip_cnts = []
        for i in range(rank):
            output_id = split_node.output[i]
            output_shape = ctx.get_shape(output_id)
            target_dtype = onnx_pb.TensorProto.INT64
            shape_i_node = ctx.make_node("Cast", [output_id], attr={"to": target_dtype},
                                         shapes=[output_shape], dtypes=[target_dtype],
                                         op_name_scope=node.name)
            trip_cnts.append(shape_i_node.output[0])
            nodes.append(shape_i_node)
        # workaround ends

        onnx_nodes = create_loop_op(ctx, node.input, true_data_type, true_data_shape, trip_cnts, rank)
        nodes.extend(onnx_nodes)
        loop_node = onnx_nodes[-1]

        val_output_id = loop_node.output[1]
    elif rank == 0:
        if_node, val_output_id = create_if_op(ctx, node.input, true_data_type, true_data_shape)
        nodes.append(if_node)

    ctx.copy_shape(node.output[0], val_output_id)
    ctx.set_dtype(node.output[0], true_data_type)

    output_node = ctx.make_node("Identity", [val_output_id], outputs=node.output,
                                shapes=[ctx.get_shape(val_output_id)], dtypes=[true_data_type])
    nodes.append(output_node)
    return nodes


# gather_input_ids is 1-D tensor, containing 3 elements:
# 0: condition data to gather on
# 1: true result to gather on
# 2: false result to father on
def create_loop_op(g, gather_input_ids, output_type, output_shape, trip_count_input_ids, rank):
    nodes = []
    cond_var_name = utils.make_name("cond_var")
    g.make_const(cond_var_name, np.array(True, dtype=np.bool))

    # Loop requires at least a variable, add a useless fake variable.
    fake_val_name = utils.make_name("fake_var")
    g.make_const(fake_val_name, np.array(0.0, dtype=np.float32))

    if rank < 1:
        raise ValueError("rank is < 1")
    trip_count_input_id = trip_count_input_ids[-1 * rank]

    loop_inputs = [trip_count_input_id,  # trip count
                   cond_var_name,  # termination condition
                   fake_val_name  # initial value of loop-carried dependencies
                  ]
    # define an extra scan output
    loop_node = g.make_node("Loop", loop_inputs, output_count=2, op_name_scope="select_loop", skip_conversion=False)
    loop_body = create_loop_body_graph(gather_input_ids, output_type, output_shape, trip_count_input_ids, rank, loop_node.name)
    loop_node.set_body_graph_as_attr("body", loop_body)
    nodes.append(loop_node)
    return nodes


def get_inputs_for_current_iteration(g, input_id, iter_index):
    nodes = []
    cond_gather_node = g.make_node("Gather", [input_id, iter_index])
    nodes.append(cond_gather_node)

    cur_cond_val_scalar_node = g.make_node("Squeeze", [cond_gather_node.output[0]], attr={"axes": [0]})
    nodes.append(cur_cond_val_scalar_node)

    return nodes, cur_cond_val_scalar_node.output[0]


def create_loop_body_graph(gather_input_ids, output_data_type, output_shape, trip_count_input_ids, rank, loop_name):
    g = Graph([], output_shapes={}, dtypes={}, target=None, opset=None, extra_opset=None, output_names=[])
    nodes = []
    iter_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    fake_var_name = utils.make_name("fake_var")
    graph_inputs = [
        utils.make_onnx_inputs_outputs(iter_name, TensorProto.INT64, (1,)),  # iteration_num
        utils.make_onnx_inputs_outputs(cond_name, TensorProto.BOOL, ()),  # condition
        utils.make_onnx_inputs_outputs(fake_var_name, TensorProto.FLOAT, ())  # loop-carried dependency
        ]
    for graph_input in graph_inputs:
        g.add_model_input(graph_input)

    # get the i'th value of condition
    cond_input_id = gather_input_ids[0]
    new_nodes, cond_input_id_for_current_iter = get_inputs_for_current_iteration(g, cond_input_id, iter_name)
    nodes.extend(new_nodes)

    # get the i'th value of true values
    true_input_id = gather_input_ids[1]
    new_nodes, true_input_id_for_current_iter = get_inputs_for_current_iteration(g, true_input_id, iter_name)
    nodes.extend(new_nodes)


    # get the i'th value of false values
    false_input_id = gather_input_ids[2]
    new_nodes, false_input_id_for_current_iter = get_inputs_for_current_iteration(g, false_input_id, iter_name)
    nodes.extend(new_nodes)

    input_ids_for_current_iter = [cond_input_id_for_current_iter, true_input_id_for_current_iter,
                                  false_input_id_for_current_iter]
    output_id = None
    rank = rank - 1
    if rank >= 1:
        nodes_1 = create_loop_op(g, input_ids_for_current_iter, output_data_type, output_shape[1:],
                                 trip_count_input_ids, rank)
        loop_1 = nodes_1[-1]
        output_id = loop_1.output[1]
        nodes.extend(nodes_1)
    elif rank == 0:
        if_node, if_node_output_id = create_if_op(g, input_ids_for_current_iter, output_data_type, output_shape[1:])
        output_id = if_node_output_id
        nodes.append(if_node)

    output_identity_name = utils.make_name("loop_output")
    loop_output_id = utils.port_name(output_identity_name)
    loop_output_node = g.make_node(
        'Identity',
        [output_id],
        outputs=[loop_output_id],
        name=output_identity_name
    )
    nodes.append(loop_output_node)

    cond_identity_name = utils.make_name("cond_output")
    cond_output_id = utils.port_name(cond_identity_name)
    identity_node = g.make_node(
        'Identity',
        [cond_name],
        outputs=[cond_output_id],
        name=cond_identity_name
    )
    nodes.append(identity_node)

    fake_var_identity_name = utils.make_name("fake_var_output")
    fake_var_output_id = utils.port_name(fake_var_identity_name)
    identity_node = g.make_node(
        'Identity',
        [fake_var_name],
        outputs=[fake_var_output_id],
        name=fake_var_identity_name
    )
    nodes.append(identity_node)
    g.set_nodes(nodes)
    g.output_names = [cond_output_id, fake_var_output_id, loop_output_id]
    g.set_dtype(cond_output_id, TensorProto.BOOL)
    g.set_shape(cond_output_id, ())
    g.set_dtype(fake_var_output_id, TensorProto.FLOAT)
    g.set_shape(fake_var_output_id, ())
    g.set_dtype(loop_output_id, output_data_type)
    g.set_shape(loop_output_id, output_shape[1:])

    return g


def create_if_op(g, input_ids, output_data_type, output_shape):
    op_name = utils.make_name("If")
    true_graph = create_body_graph_for_if_branch(output_data_type, output_shape, input_ids[1], op_name)
    false_graph = create_body_graph_for_if_branch(output_data_type, output_shape, input_ids[2], op_name)
    out_name = port_name(op_name)

    # output a scalar
    if_node = g.make_node("If", [input_ids[0]], outputs=[out_name], name=op_name, skip_conversion=False)
    if_node.set_body_graph_as_attr("then_branch", true_graph)
    if_node.set_body_graph_as_attr("else_branch", false_graph)
    return if_node, out_name


def create_body_graph_for_if_branch(data_type, output_shape, chosen_cur_cond_val_out_name, op_name):
    g = Graph([], output_shapes={}, dtypes={}, target=None, opset=None, extra_opset=None, output_names=[])
    nodes = []
    name = utils.make_name("Identity")
    identity_node = g.make_node(
        'Identity',
        inputs=[chosen_cur_cond_val_out_name],
        outputs=['y'],
        name=name
    )
    nodes.append(identity_node)

    g.output_names = ["y"]
    g.set_dtype("y", data_type)
    g.set_shape("y", output_shape)

    g.set_nodes(nodes)

    return g
