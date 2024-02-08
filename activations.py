


from base import Base_Layer
import cupy as cp
import numpy as np


class Activation(Base_Layer):
    def __init__(self):
        pass
    
    def get_next_id2(self):
        return super().get_next_id()

    def forward(self):
        pass

    def backward(self):
        pass
    




class Tanh(Activation):
   def __init__(self,device='cpu'):
         
       # main device     
       self.device=device
       self.id = f"{self.__class__.__name__}_{super().get_next_id2()}"
   
   def set_gpu(self):
       if self.device == 'gpu':    
           print('Tanh is already on gpu')
       else:
           self.device = 'gpu' 
             
   def set_cpu(self):
      if self.device == 'cpu':
          print('Tanh is already on cpu')
      else:
          self.device = 'cpu'        

   def forward(self, input_):    
       if self.device=='gpu':
          self.inp=input_   
          return cp.tanh(self.inp)
       else:
          self.inp=input_   
          return np.tanh(self.inp)

   def backward(self, output_gradient, learning_rate):
       if self.device=='gpu':
          temp = 1-cp.tanh(self.inp) ** 2
          return cp.multiply(cp.asarray(output_gradient),temp)
       else:
          temp = 1 - np.tanh(self.inp) ** 2   
          return np.multiply(output_gradient,temp) 

   def state_dict(self):
        return {"activation": type(self).__name__,"device":self.device}
           
   def load_state_dict(self):
       pass 
   





# under development =============================================================================>>>>>




class Sigmoid(Activation):

    def __init__(self,device='cpu'):
        # main device     
        self.device=device
        self.id = f"{self.__class__.__name__}_{super().get_next_id2()}"
             
    def set_gpu(self):
        if self.device == 'gpu':
            print('Sigmoid is already on gpu')
        else:
            self.device = 'gpu' 
             
    def set_cpu(self):
        if self.device == 'cpu':
            print('Sigmoid is already on cpu')
        else:
            self.device = 'cpu'
        
    def forward(self, input_):
        if self.device=='gpu':
           # could change that and move it to gpu (self.inp=input_)
           
           #self.inp=input_   
           self.inp=cp.asarray(input_) 
           self.outp= 1 / (1 + cp.exp(-self.inp))
           #return self.outp
           return cp.asarray(self.outp)
        else:
           self.inp=input_   
           self.outp= 1 / (1 + np.exp(-self.inp))
           return self.outp

    def backward(self, output_gradient, learning_rate):
        if self.device == 'gpu':
           # test only
           #print(f'type of (output_gradient) is: | {type(output_gradient)}')
           #print(f'type of (self.inp) is: | {type(self.inp)}')
           self.a_tmp= 1 / (1 + cp.exp(-self.inp))
           # test only
           #print(f'type of (self.b_tmp) is: | {type(self.a_tmp)}')
           
           self.b_tmp= self.a_tmp * (1-self.a_tmp)
           # test only
           #print(f'type of (self.b_tmp) is: | {type(self.b_tmp)}')
           return cp.multiply(cp.asarray(output_gradient),self.b_tmp)
        else:
           self.a_tmp= 1 / (1 + np.exp(-self.inp)) 
           self.b_tmp= self.a_tmp * (1-self.a_tmp)
           return np.multiply(output_gradient,self.b_tmp)

    def state_dict(self):
        return {"activation": type(self).__name__,"device":self.device}
           
    def load_state_dict(self):
        pass
    




class Softmax(Base_Layer):

    def __init__(self,device='cpu'):
        # main device     
        self.device=device
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"    
    
    def set_gpu(self):
        if self.device == 'gpu':
            print('Softmax is already on gpu')
        else:
            self.device = 'gpu' 
             
    def set_cpu(self):
        if self.device == 'cpu':
            print('Softmax is already on cpu')
        else:
            self.device = 'cpu'

    def forward(self, input_):
       if self.device=='gpu':
          self.inp=cp.exp(input_)   
          self.outp=self.inp / cp.sum(self.inp)
          return self.outp
       else:
          self.inp = np.exp(input_)
          self.outp = self.inp / np.sum(self.inp)
          return self.outp
       
    def backward(self, output_gradient, learning_rate):
       if self.device=='gpu':
          self.n = cp.size(self.outp)  # Get the size using CuPy's size function
          return cp.dot(cp.multiply(cp.identity(self.n) - self.outp.T, self.outp), cp.asarray(output_gradient))
       else:
          self.n = np.size(self.outp)
          return np.dot((np.identity(self.n) - self.outp.T) * self.outp, output_gradient)
    
    def state_dict(self):
        return {"activation": type(self).__name__,"device":self.device}
           
    def load_state_dict(self):
        pass
    





class ReLU(Activation):
   def __init__(self, device='cpu'):
       super().__init__()
       self.device = device
       self.id = f"{self.__class__.__name__}_{super().get_next_id2()}"
   
   def set_gpu(self):
       if self.device == 'gpu':    
           print('ReLU is already on gpu')
       else:
           self.device = 'gpu' 
             
   def set_cpu(self):
      if self.device == 'cpu':
          print('ReLU is already on cpu')
      else:
          self.device = 'cpu'        

   def forward(self, input_):
       if self.device == 'gpu':
           self.inp = cp.asarray(input_)
           return cp.maximum(self.inp, 0)
       else:
           self.inp = input_
           return np.maximum(self.inp, 0)

   def backward(self, output_gradient, learning_rate):
       if self.device == 'gpu':
           temp = cp.asarray(self.inp > 0)
           return cp.multiply(cp.asarray(output_gradient), temp)
       else:
           temp = self.inp > 0
           return np.multiply(output_gradient, temp)

   def state_dict(self):
        return {"activation": type(self).__name__,"device":self.device}
           
   def load_state_dict(self):
       pass 














