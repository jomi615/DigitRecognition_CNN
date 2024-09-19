import numpy as np

#Reference: https://www.digitalocean.com/community/tutorials/relu-function-in-python
def rel_activate(input):
    if input > 0:
        return input
    else:
        return 0

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    activated = np.vectorize(rel_activate)
    output['data'] = activated(input_data['data'])
    return output



def relu_backward(output, input_data, layer):
    # Initialize the input_od with zeros of the same shape as input_data
    relu = np.maximum(input_data['data'], 0)

    # Each element of applied is True where the corresponding element in relu is equal
    # to the corresponding element in input_data['data'], and False otherwise.
    applied = np.equal(relu, input_data['data'])

    # Multiply the output.diff by the mask (temp) to calculate input_od
    input_od = output['diff'] * applied

    return input_od
