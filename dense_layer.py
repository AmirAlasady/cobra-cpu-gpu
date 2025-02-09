


from base import Base_Layer
import cupy as cp # type: ignore
import numpy as np

class Dense(Base_Layer):
    def __init__(self, input_size, output_size, name=None, initialization='xavier', device='cpu'):
        super().__init__()
        self.name = name
        self.device = device
        self.initialization = initialization
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
        # Weight initialization with device control
        if self.device == 'gpu':
            array_module = cp
            init_std = 0.01
            if initialization == 'xavier':
                init_std = cp.sqrt(2.0 / (input_size + output_size))
            self.weights = array_module.random.randn(output_size, input_size).astype(cp.float32) * init_std
            self.bias = cp.zeros((output_size, 1), dtype=cp.float32)
        else:
            array_module = np
            init_std = 0.01
            if initialization == 'xavier':
                init_std = np.sqrt(2.0 / (input_size + output_size))
            self.weights = array_module.random.randn(output_size, input_size).astype(np.float32) * init_std
            self.bias = np.zeros((output_size, 1), dtype=np.float32)
            
        self.weights_grad = None
        self.bias_grad = None

    def set_gpu(self):
        if self.device != 'gpu':
            self.weights = cp.asarray(self.weights)
            self.bias = cp.asarray(self.bias)
            self.device = 'gpu'

    def set_cpu(self):
        if self.device != 'cpu':
            self.weights = cp.asnumpy(self.weights)
            self.bias = cp.asnumpy(self.bias)
            self.device = 'cpu'

    def forward(self, inputs):
        self.inputs = inputs
        if self.device == 'gpu':
            return cp.dot(self.weights, self.inputs) + self.bias
        return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_gradient):
        batch_size = output_gradient.shape[1]
        
        if self.device == 'gpu':
            self.weights_grad = cp.dot(output_gradient, self.inputs.T) / batch_size
            self.bias_grad = cp.mean(output_gradient, axis=1, keepdims=True)
            return cp.dot(self.weights.T, output_gradient)
        
        self.weights_grad = np.dot(output_gradient, self.inputs.T) / batch_size
        self.bias_grad = np.mean(output_gradient, axis=1, keepdims=True)
        return np.dot(self.weights.T, output_gradient)

    def parameters(self):
        return [
            (self.weights, self.weights_grad),
            (self.bias, self.bias_grad)
        ]

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
        #self.initialization = state_dict.get('initialization', 'xavier')
