


from abc import abstractmethod
from base import Base_Layer
import cupy as cp # type: ignore
import numpy as np

class Activation(Base_Layer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def set_gpu(self):
        self.device = 'gpu'

    def set_cpu(self):
        self.device = 'cpu'

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, output_gradient):
        pass

class Tanh(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.inputs = None
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
    def forward(self, inputs):
        self.inputs = inputs
        if self.device == 'gpu':
            return cp.tanh(inputs)
        return np.tanh(inputs)

    def backward(self, output_gradient):
        if self.device == 'gpu':
            return output_gradient * (1 - cp.tanh(self.inputs)**2)
        return output_gradient * (1 - np.tanh(self.inputs)**2)

    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []
    
class ReLU(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.inputs = None
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        self.inputs = inputs
        if self.device == 'gpu':
            return cp.maximum(0, inputs)
        return np.maximum(0, inputs)

    def backward(self, output_gradient):
        if self.device == 'gpu':
            return output_gradient * (self.inputs > 0).astype(cp.float32)
        return output_gradient * (self.inputs > 0).astype(float)

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []
    
class Sigmoid(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.outputs = None
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        if self.device == 'gpu':
            self.outputs = 1 / (1 + cp.exp(-inputs))
        else:
            self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, output_gradient):
        if self.device == 'gpu':
            return output_gradient * (self.outputs * (1 - self.outputs))
        return output_gradient * (self.outputs * (1 - self.outputs))

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass
    def parameters(self):
        return []
    
class Softmax(Activation):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.outputs = None
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        if self.device == 'gpu':
            exp = cp.exp(inputs - cp.max(inputs, axis=0))
            self.outputs = exp / cp.sum(exp, axis=0)
        else:
            exp = np.exp(inputs - np.max(inputs, axis=0))
            self.outputs = exp / np.sum(exp, axis=0)
        return self.outputs
    def parameters(self):
        return []
    def backward(self, output_gradient):
        # Assumes combined with cross-entropy loss
        return output_gradient

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []
    
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01, device='cpu'):
        super().__init__(device)
        self.alpha = alpha
        self.inputs = None
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        self.inputs = inputs
        if self.device == 'gpu':
            return cp.where(inputs > 0, inputs, self.alpha * inputs)
        return np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, output_gradient):
        if self.device == 'gpu':
            return output_gradient * cp.where(self.inputs > 0, 1, self.alpha)
        return output_gradient * np.where(self.inputs > 0, 1, self.alpha)

    def parameters(self):
        return []
    
    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass

class ELU(Activation):
    def __init__(self, alpha=1.0, device='cpu'):
        super().__init__(device)
        self.alpha = alpha
        self.inputs = None
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        self.inputs = inputs
        if self.device == 'gpu':
            return cp.where(inputs > 0, inputs, self.alpha * (cp.exp(inputs) - 1))
        return np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))

    def backward(self, output_gradient):
        if self.device == 'gpu':
            return output_gradient * cp.where(self.inputs > 0, 1, self.alpha * cp.exp(self.inputs))
        return output_gradient * np.where(self.inputs > 0, 1, self.alpha * np.exp(self.inputs))

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []