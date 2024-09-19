import numpy as np
from utils import im2col_conv, col2im_conv, im2col_conv_batch
import numpy as np 
def conv_layer_forward(input_data, layer, param):
    """
    Forward pass for a convolutional layer.

    Parameters:
    - input_data (dict): A dictionary containing the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - param (dict): A dictionary containing the parameters 'b' and 'w'.
    """
    h_in = input_data['height']
    w_in = input_data['width']
    c = input_data['channel']
    batch_size = input_data['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']
    num = layer['num']

    # resolve output shape
    h_out = (h_in + 2*pad - k) // stride + 1
    w_out = (w_in + 2*pad - k) // stride + 1

    assert h_out == int(h_out), 'h_out is not integer'
    assert w_out == int(w_out), 'w_out is not integer'

    input_n = {
        'height': h_in,
        'width': w_in,
        'channel': c,
        'data': input_data['data'],
        'batch_size': batch_size
    }

    W = param['w']
    bias = param['b']
    conv_batch = im2col_conv_batch(input_n, layer, h_out, w_out)  
    res = np.zeros((num,h_out,w_out))

    #Reference: https://canvas.sfu.ca/courses/80521/discussion_topics/1694509
    for b in range(batch_size):
        img = conv_batch.reshape(c * k * k, h_out * w_out * batch_size)
        # print(img.shape)
        res = np.dot(img.T, W) + bias
        trans_res = np.transpose(res)
        #print(trans_res.shape)
        trans_res = trans_res.reshape((num * h_out * w_out, batch_size))
        
    output = {
        'height': h_out,
        'width': w_out,
        'channel': num,
        'batch_size': batch_size,
        'data':trans_res # Initialize data with zeros
    }
    return output


def conv_layer_backward(output, input_data, layer, param):
    """
    Compute the backward pass for the convolution layer.
    
    Parameters:
    - output (dict): A dictionary containing the output of the forward pass.
    - input_data (dict): A dictionary containing the original input to the forward function.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - param (dict): A dictionary containing the parameters 'b' and 'w'.

    Returns:
    - param_grad (dict): A dictionary containing the gradients with respect to the parameters 'b' and 'w'.
    - input_od (numpy.ndarray): The gradients with respect to the input.
    """
    
    h_in = input_data['height']
    w_in = input_data['width']
    c = input_data['channel']
    batch_size = input_data['batch_size']
    k = layer['k']
    group = layer['group']
    num = layer['num']

    h_out = output['height']
    w_out = output['width']
    input_n = {'height': h_in, 'width': w_in, 'channel': c}
    
    input_od = np.zeros(input_data['data'].shape)
    param_grad = {'b': np.zeros(param['b'].shape), 'w': np.zeros(param['w'].shape)}

    for n in range(batch_size):
        input_n['data'] = input_data['data'][:, n]
        col = im2col_conv(input_n, layer, h_out, w_out)
        col = np.reshape(col, (k*k*c, h_out*w_out), order='F')
        col_diff = np.zeros(col.shape)
        temp_data_diff = np.reshape(output['diff'][:, n], (h_out*w_out, num), order='F')
        
        for g in range(group):
            g_c_idx = slice(g*k*k*c//group, (g+1)*k*k*c//group)
            g_num_idx = slice(g*num//group, (g+1)*num//group)
            col_g = col[g_c_idx, :]
            weight = param['w'][:, g_num_idx]
            
            # get the gradient of param
            param_grad['b'][:, g_num_idx] += np.sum(temp_data_diff[:, g_num_idx], axis=0)
            param_grad['w'][:, g_num_idx] += col_g.dot(temp_data_diff[:, g_num_idx])
            col_diff[g_c_idx, :] = weight.dot(temp_data_diff[:, g_num_idx].T)
            
        im = col2im_conv(col_diff.ravel(order='F'), input_data, layer, h_out, w_out)
        # set the gradient w.r.t to input.data
        input_od[:, n] = im.ravel(order='F')

    return param_grad, input_od

