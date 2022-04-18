# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 14
import torch

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args


# Note [ONNX operators that are added/updated in opset 14]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New operators:
#   HardSwish, Trilu
#
# Updated operators:
#   Reshape
#   Add, Sub, Mul, Div
#   GRU, LSTM, RNN
#   BatchNorm, Cumsum, Relu

@parse_args("v")
def hardswish(g, self):
    return g.op("HardSwish", self)


@parse_args("v", "i")
def tril(g, self, diagonal, out=None):
    k = g.op("Constant", value_t=torch.tensor(diagonal, dtype=torch.int64))
    return g.op("Trilu", self, k, upper_i=0)


@parse_args("v", "i")
def triu(g, self, diagonal, out=None):
    k = g.op("Constant", value_t=torch.tensor(diagonal, dtype=torch.int64))
    return g.op("Trilu", self, k, upper_i=1)


@parse_args("v", "v")
def reshape(g, self, shape):
    return sym_help._reshape_helper(g, self, shape)


@parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    sym_help.check_training_mode(training, "batch_norm")
    weight, bias, running_mean, running_var = sym_help._batchnorm_helper(g, input, weight, bias, running_mean,
                                                                         running_var)
    out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
               epsilon_f=eps,
               momentum_f=1 - momentum,
               training_mode_i=0 if not training else 1,
               outputs=1 if not training else 3)
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        return res


@parse_args("v", "v", "i", "i", 'b')
def grid_sampler(g, input, grid,
                 mode_enum: int, padding_mode_enum: int, align_corners: bool):
    if mode_enum == 0:
        mode = "bilinear"
    elif mode_enum == 1:
        mode = "nearest"
    else:
        mode = "bicubic"

    if padding_mode_enum == 0:
        padding_mode = "zeros"
    elif padding_mode_enum == 1:
        padding_mode = "border"
    else:
        padding_mode = "reflection"

    return g.op("GridSample", input, grid,
                mode_s=mode, padding_mode_s=padding_mode, align_corners_i=int(align_corners))


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def deform_conv2d(g,
                  input,
                  weight,
                  offset,
                  mask,
                  bias,
                  stride_h,
                  stride_w,
                  pad_h,
                  pad_w,
                  dil_h,
                  dil_w,
                  n_weight_grps,
                  n_offset_grps,
                  use_mask):
    return g.op("DeformableConvolution", input, offset, weight, bias,
                dilations_i=(dil_h, dil_w), strides_i=(stride_h, stride_w))


def chunk(g, self, chunks, dim):
    """
    Fix non-int chunk_size of opset11
    """
    from torch.onnx.symbolic_opset9 import floor, expand
    from torch.onnx.symbolic_opset11 import split
    # Calculate chunk size for dynamic chunk
    dim_size = g.op("Gather", g.op("Shape", self), dim, axis_i=0)
    chunk_size_s = g.op("Sub", chunks, g.op("Constant", value_t=torch.tensor([1], dtype=torch.long)))
    chunk_size = g.op("Div", g.op("Add", dim_size, chunk_size_s), chunks)
    chunk_size = floor(g, chunk_size)
    # Create splits vector
    chunk_vec = [expand(g, chunk_size, chunk_size_s, None),
                 g.op("Sub", dim_size, g.op("Mul", chunk_size, chunk_size_s))]
    chunk_vec = g.op("Concat", *chunk_vec, axis_i=0)
    return split(g, self, chunk_vec, dim)
