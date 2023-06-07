
import torch
from .motion import dense_image_warp

def execute_flow_warp(data, u, v):
    def warp_batch(data, u, v):
        data = data[..., None] # [M, 256, 256, 1]
        flow = torch.cat([u[...,None], v[...,None]], 3)
        if('complex' in str(data.dtype)):
            data_real = data.real
            data_imag = data.imag
            output_real = dense_image_warp(data_real, flow)[...,0] # [M, 256, 256]
            output_imag = dense_image_warp(data_imag, flow)[...,0] # [M, 256, 256]
            output = torch.complex(output_real, output_imag)
        else:
            output = dense_image_warp(data, flow)[...,0] # [M, 256, 256]
        return output
    shape = tf_shape(data)
    mf_shape = tf_shape(u)
    data = torch.reshape(data, [shape[0]*shape[1], 1] + shape[2:4])[:,0]
    u = torch.reshape(u, [mf_shape[0]*mf_shape[1], 1] + mf_shape[2:4])[:,0]
    v = torch.reshape(v, [mf_shape[0]*mf_shape[1], 1] + mf_shape[2:4])[:,0]

    new_data = warp_batch(data, u, v)
    new_data = torch.reshape(new_data, shape)
    return new_data

def tf_shape(data):
    shape = data.shape
    shape = [shape[i] for i in range(len(data.shape))]
    return shape

