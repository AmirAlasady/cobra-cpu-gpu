from abc import ABC, abstractmethod


class Base_Layer(ABC):
    _id_counter = 0

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.device = 'cpu'  # Default device
        self.id = f"{self.__class__.__name__}_{self.get_next_id()}"

    def get_next_id(self):
        Base_Layer._id_counter += 1
        return Base_Layer._id_counter

    @abstractmethod
    def set_gpu(self):
        pass
    
    @abstractmethod
    def set_cpu(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, output_gradient):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
