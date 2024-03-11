





from sequential import Sequential
from dlayer import Layer
from activations import *
from loss import loss_base,Loss_on_cpu,Loss_on_gpu
from abc import abstractmethod
import matplotlib.pyplot as plt

# base custom model class 
class Model:
    
    # genaric base setter 
    def set(self):
        if self.device == 'gpu':

            for name, param in self.__dict__.items():
                if name == 'device':
                    pass
                else:
                    if param.device == 'gpu':
                       print(f'{name}: is already on <gpu> in: {type(self).__name__}')
                    else:
                        param.set_gpu()
                        print(f'<{name}> of type <{param}> : moved to (gpu)')
        else:

            for name, param in self.__dict__.items():
                if name == 'device':
                    pass
                else:
                    if param.device == 'cpu':
                       print(f'{name}: is already on <cpu> in: {type(self).__name__}')
                    else:
                        param.set_cpu()
                        print(param.device)
                        print(f'<{name}> of type <{param}> : moved to (cpu)')
        
             
    # set_gpu
    @abstractmethod
    def set_gpu(self):
        if self.device == 'gpu':
            print(f'{type(self).__name__} is already on gpu')
        else:
            self.device = 'gpu'
            for name, param in self.__dict__.items():               
                if name == 'device':
                    pass
                else:
                    param.set_gpu()
            print(f'moved to: <gpu>')
                           
    # set_cpu
    @abstractmethod
    def set_cpu(self):
        if self.device == 'cpu':
            print(f'{type(self).__name__} is already on cpu')
        else:
            self.device = 'cpu'
            for name, param in self.__dict__.items():
                
                if name == 'device':
                    pass
                else:
                    param.set_cpu()
            print(f'moved to: <cpu>')
    
    # required forward method
    @abstractmethod
    def forward(self,input_data):
        self.input_data=input_data
        for name, param in self.__dict__.items(): 
            if name == 'device':
               pass
            else:         
                if isinstance(param, Layer):
                    self.input_data = param.forward(self.input_data)
                if isinstance(param, Sequential):
                    self.input_data = param.forward(self.input_data)
                if isinstance(param, Tanh):
                    self.input_data = param.forward(self.input_data)
                if isinstance(param, Sigmoid):
                    self.input_data = param.forward(self.input_data)
                if isinstance(param, Softmax):
                    self.input_data = param.forward(self.input_data)
                if isinstance(param, ReLU):
                    self.input_data = param.forward(self.input_data)
                    
        return self.input_data
    
    # required backward method
    @abstractmethod
    def backward(self,grad,lr):
        self.grad=grad
        for name, param in reversed(list(self.__dict__.items())):
            if name == 'device':
               pass
            else:
                if isinstance(param, Layer):
                    self.grad= param.backward(self.grad,lr)
                if isinstance(param, Sequential):
                    self.grad= param.backward(self.grad,lr)
                if isinstance(param, Tanh):
                    self.grad= param.backward(self.grad,lr)
                if isinstance(param, Sigmoid):
                    self.grad= param.backward(self.grad,lr)
                if isinstance(param, Softmax):
                    self.grad= param.backward(self.grad,lr)
                if isinstance(param, ReLU):
                    self.grad= param.backward(self.grad,lr)
        return self.grad
        
    

    # experimental space
    
    def train(self,epochs,X,Y,lr,loss,loss_prime,plot=True):
        self.loss_history=[]
        self.plot=plot
        for i in range(epochs):
            self.error=0
            for x,y in zip(X,Y):
                self.output=self.forward(x)
                self.error += loss(y,self.output)           
                self.grad = loss_prime(y,self.output)
                self.grad=self.backward(self.grad,lr)

            self.error /= len(x)
            self.loss_history.append(self.error)
            print(f"{i + 1}/{epochs}, error={self.error}")
            
        if self.plot:
            if self.device=='gpu':
                self.loss_history2 = [item.item() for item in self.loss_history]
                plt.plot(self.loss_history2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss During Training')
                plt.show()
            else:
                 plt.plot(self.loss_history)
                 plt.xlabel('Epoch')
                 plt.ylabel('Loss')
                 plt.title('Loss During Training')
                 plt.show()
                 
    # super state_dict saver 'for all constructor parameters'
    def state_dict(self):
        state_dict  = {}
        for name, param in self.__dict__.items():
            if name == 'device':
               pass
            else:
                #state_dict[name] = param.state_dict()
                
                if isinstance(param, Sequential):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Layer):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Tanh):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Sigmoid):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Softmax):                   
                   state_dict[name] = param.state_dict()
                if isinstance(param, ReLU):                   
                   state_dict[name] = param.state_dict()  
                   
        state_dict["device"] = self.device
        return state_dict
    # super state_dict loader 'for all constructor parameters'
    def load_state_dict(self, state_dict):
        if state_dict['device']=='gpu':
           for name, param in self.__dict__.items():
               if name == 'device':
                  pass
               else:
                   if isinstance(param, Sequential):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Layer):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Tanh):  
                       param.load_state_dict()
                   if isinstance(param, Sigmoid):  
                       param.load_state_dict()
                   if isinstance(param, Softmax):  
                       param.load_state_dict()
                   if isinstance(param, ReLU):  
                       param.load_state_dict()
           self.set_gpu()
        else:
           for name, param in self.__dict__.items():
               if name == 'device':
                  pass
               else:
                   if isinstance(param, Sequential):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Layer):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Tanh):  
                       param.load_state_dict()
                   if isinstance(param, Sigmoid):  
                       param.load_state_dict()
                   if isinstance(param, Softmax):  
                       param.load_state_dict()
                   if isinstance(param, ReLU):  
                       param.load_state_dict()
           self.set_cpu()