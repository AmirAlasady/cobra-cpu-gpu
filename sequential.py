from dlayer import Layer
import matplotlib.pyplot as plt
class Sequential:

    # list of Layer objects that we will loop on and go forward or backward while saving output per epoch and asign it as input in epoch+1
    def __init__(self,network:list,device:str='cpu'):
        self.network_=network
        assert device in ['cpu', 'gpu']
        self.device=device

        # modular code auto setter to parameter device if gpu
        if self.device == 'gpu':
            for layer_element in self.network_:
               layer_element.set_gpu()
         
        # modular code auto setter to parameter device if cpu
        else:
           for layer_element in self.network_:
                layer_element.set_cpu()
               
    def set_gpu(self):
      # check if already on gpu
      if self.device == 'gpu':
        print('Sequential is already on gpu')
      else:
        # sets list layer objects to gpu  
        self.device = 'gpu'
        for layer_element in self.network_:
           layer_element.set_gpu()
        print('moved to gpu!')
    def set_cpu(self):  
      # check if already on cpu
      if self.device == 'cpu':
        print('Sequential is already on cpu')
      else:
       # sets list layer objects to cpu  
        self.device = 'cpu'
        for layer_element in self.network_:
           layer_element.set_cpu()
        print('moved to cpu!')
        
    # forward taking single input and moving forward through the hall network and each output from nth layer is the next input for the next layer "per epoch"
    def forward(self,input_):
        self.output = input_
        for layer in self.network_:
            self.output=layer.forward(self.output)
        return self.output

    # back propagation from the end of the network to the first layer , starting from taking a grad from loss grads then learning rate then back prop fore each layer in reverse 'computing grad of inputs and parameters of nth layer the passing inputs as output grad for next layer for the hall list' 
    def backward(self,grad,lr):
        self.grad=grad
        for layer in self.network_[::-1]:
            self.grad=layer.backward(self.grad,lr)
        return self.grad
    
    


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


    # saving the state dictionary for all layers in the list
    def state_dict(self):
        model_state_dict = {}
        for i, layer in enumerate(self.network_):
            layer_id = layer.id  # Use the unique layer ID for keys
            model_state_dict[layer_id] = layer.state_dict()
        # Append the "device" key-value pair at the end
        model_state_dict["device"] = self.device

        return model_state_dict
    
    # loading the state dictionary for all layers in the list
    def load_state_dict(self, state_dict):
        
        # set state dict overwrite device attribute 
        if state_dict['device']=='gpu':
           keys = state_dict.keys()
           keys=list(keys)
           for i, layer in enumerate(self.network_):
               if isinstance(layer, Layer):
                   layer_id = layer.id
                   key=keys[i]
                   layer.load_state_dict(state_dict[key])
               else:
                   pass
           self.set_gpu()
        else:
           keys = state_dict.keys()
           keys=list(keys)
           for i, layer in enumerate(self.network_):
               if isinstance(layer, Layer):
                   layer_id = layer.id
                   key=keys[i]
                   layer.load_state_dict(state_dict[key])
               else:
                   pass
           self.set_cpu()