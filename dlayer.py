


from base import Base_Layer
import cupy as cp
import numpy as np


# Layer class gpu support
class Layer(Base_Layer):

    def __init__(self, input_size, output_size, name=None, device='cpu'):
      # Check if the device is valid
      assert device in ['cpu', 'gpu']
      # Store the device
      self.device = device
      # inputs chock set
      if self.device=='gpu':
        self.weights = cp.random.randn(output_size, input_size)
        self.bias = cp.random.randn(output_size, 1)
      else:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
      # Assign ID for state_dict usage
      self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
      self.name = name
    def set_gpu(self):
      # check if already on gpu
      if self.device == 'gpu':
        print('Layer is already on gpu')
      else:
        # Move weights and bias to GPU
        self.weights = cp.asarray(self.weights)
        self.bias = cp.asarray(self.bias)
        self.device = 'gpu'

    def set_cpu(self):  
      # check if already on cpu
      if self.device == 'cpu':
        print('Layer is already on cpu')
      else:
        # Move weights and bias to CPU
        self.weights = cp.asnumpy(self.weights)
        self.bias = cp.asnumpy(self.bias)
        self.device = 'cpu'

    def forward(self, inputs):
      self.inputs = inputs
      if self.device == 'gpu':
        # Move inputs to GPU
        #self.inputs = cp.asarray(self.inputs)
        # return operation done on gpu
        return cp.dot(self.weights, self.inputs) + self.bias
      else:
        # return operation done on cpu
        return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_gradient, learning_rate):
      
      if self.device == 'gpu':
        #output_gradient = cp.asarray(output_gradient)
        learning_rate = learning_rate
        weights_gradient = cp.dot( output_gradient, self.inputs.T)
        input_gradient = cp.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate *output_gradient
        return input_gradient
      else:
        weights_gradient = np.dot(output_gradient, self.inputs.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient



    def state_dict(self):
      if self.device == 'gpu':
        return {
            "weights": cp.asnumpy(self.weights),
            "bias": cp.asnumpy(self.bias),
            "device": self.device
        }
      else:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "device": self.device
        }

    def load_state_dict(self, state_dict):
      if state_dict['device'] == 'gpu':
        self.weights = cp.asarray(state_dict['weights'])
        self.bias = cp.asarray(state_dict['bias'])
        self.device = state_dict['device']
        self.set_gpu()  #extra check
      else:
        self.weights = state_dict['weights']
        self.bias = state_dict['bias']
        self.device = state_dict['device']
        self.set_cpu()  #extra check
        


