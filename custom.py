from sequential import Sequential
from dense_layer import Dense  # Your Dense layer class
from activations import ELU, LeakyReLU, Tanh, Sigmoid, Softmax, ReLU
import matplotlib.pyplot as plt
import numpy as np  # Used for reshaping, etc.

class Model:
    def __init__(self, device='cpu'):
        # Initialize with a device flag. All submodules should be added as attributes.
        self.device = device

    def set(self):
        """
        Generic setter that iterates through sub-components and sets their device.
        """
        for name, param in self.__dict__.items():
            if name == 'device':
                continue
            if hasattr(param, "device"):
                if param.device == self.device:
                    print(f"{name} is already on <{self.device}> in {type(self).__name__}")
                else:
                    if self.device == 'gpu':
                        param.set_gpu()
                        print(f"<{name}> of type {type(param).__name__} moved to gpu")
                    else:
                        param.set_cpu()
                        print(f"<{name}> of type {type(param).__name__} moved to cpu")

    def set_gpu(self):
        """
        Switch the model and its sub-components to GPU.
        """
        if self.device == 'gpu':
            print(f"{type(self).__name__} is already on gpu")
        else:
            self.device = 'gpu'
            for name, param in self.__dict__.items():
                if name == 'device':
                    continue
                if hasattr(param, "set_gpu"):
                    param.set_gpu()
            print("Model and its sub-components have been moved to gpu")

    def set_cpu(self):
        """
        Switch the model and its sub-components to CPU.
        """
        if self.device == 'cpu':
            print(f"{type(self).__name__} is already on cpu")
        else:
            self.device = 'cpu'
            for name, param in self.__dict__.items():
                if name == 'device':
                    continue
                if hasattr(param, "set_cpu"):
                    param.set_cpu()
            print("Model and its sub-components have been moved to cpu")

    def forward(self, input_data):
        """
        Propagate input_data through all sub-components.  
        The order is determined by iterating over __dict__.
        """
        self.input_data = input_data
        for name, param in self.__dict__.items():
            if name == 'device':
                continue
            # Check for known layer/activation types.
            if isinstance(param, (Dense, Sequential, Tanh, Sigmoid, Softmax, ReLU)):
                self.input_data = param.forward(self.input_data)
        return self.input_data

    def backward(self, grad):
        """
        Backpropagate the gradient through the network in reverse order.
        """
        self.grad = grad
        for name, param in reversed(list(self.__dict__.items())):
            if name == 'device':
                continue
            if isinstance(param, (Dense, Sequential, Tanh, Sigmoid, Softmax, ReLU)):
                self.grad = param.backward(self.grad)
        return self.grad

    def train(self, epochs, X, Y, loss_fn, loss_prime, optimizer=None, plot=False):
        """
        Train the model.
          - X, Y: iterables (or lists) of input samples and labels.
          - loss: loss function.
          - loss_prime: derivative of the loss function.
          - optimizer: an optimizer instance (e.g., SGD, Adam, RMSprop) that implements step(layers).
          - plot: if True, displays a live plot of the loss.
        """
        self.loss_history = []

        # Gather all trainable layers (here, Dense layers) from the model.
        trainable_layers = []
        for name, param in self.__dict__.items():
            if isinstance(param, Dense):
                trainable_layers.append(param)
            elif isinstance(param, Sequential):
                # Recursively extract Dense layers from a Sequential.
                def extract_layers(layer):
                    layers = []
                    if isinstance(layer, Dense):
                        layers.append(layer)
                    elif isinstance(layer, Sequential):
                        for l in layer.layers:
                            layers.extend(extract_layers(l))
                    return layers
                trainable_layers.extend(extract_layers(param))

        # If plotting is enabled, initialize an interactive plot.
        if plot:
            plt.ion()
            fig, ax = plt.subplots()
            line, = ax.plot([], [])  # Create an empty line.
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss During Training')
            ax.grid(True)
            plt.show(block=False)

        for epoch in range(epochs):
            total_loss = 0
            # Process each training sample.
            for x, y in zip(X, Y):
                # Ensure x and y are column vectors.
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                output = self.forward(x)
                total_loss += loss_fn(y, output)
                grad = loss_prime(y, output)
                self.backward(grad)

                # Use the optimizer to update parameters, if provided.
                if optimizer is not None:
                    optimizer.step(trainable_layers)

            # Compute average loss over the training samples.
            avg_loss = total_loss / len(X)
            self.loss_history.append(avg_loss)
            print(f"{epoch + 1}/{epochs}, error={avg_loss}")

            # Update the plot.
            if plot:
                line.set_xdata(range(len(self.loss_history)))
                line.set_ydata(self.loss_history)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        if plot:
            plt.ioff()
            plt.show()

    def state_dict(self):
        state_dict  = {}
        for name, param in self.__dict__.items():
            if name == 'device':
               pass
            else:
                #state_dict[name] = param.state_dict()
                
                if isinstance(param, Sequential):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Dense):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Tanh):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Sigmoid):  
                   state_dict[name] = param.state_dict()
                if isinstance(param, Softmax):                   
                   state_dict[name] = param.state_dict()
                if isinstance(param, ReLU):                   
                   state_dict[name] = param.state_dict() 
                if isinstance(param, LeakyReLU):                   
                   state_dict[name] = param.state_dict()
                if isinstance(param, ELU):                   
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
                   if isinstance(param, Dense):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Tanh):  
                       param.load_state_dict()
                   if isinstance(param, Sigmoid):  
                       param.load_state_dict()
                   if isinstance(param, Softmax):  
                       param.load_state_dict()
                   if isinstance(param, ReLU):  
                       param.load_state_dict()
                   if isinstance(param, LeakyReLU):
                       param.load_state_dict()
                   if isinstance(param, ELU):
                       param.load_state_dict()
                                           
           self.set_gpu()
        else:
           for name, param in self.__dict__.items():
               if name == 'device':
                  pass
               else:
                   if isinstance(param, Sequential):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Dense):  
                       param.load_state_dict(state_dict[name])
                   if isinstance(param, Tanh):  
                       param.load_state_dict()
                   if isinstance(param, Sigmoid):  
                       param.load_state_dict()
                   if isinstance(param, Softmax):  
                       param.load_state_dict()
                   if isinstance(param, ReLU):  
                       param.load_state_dict()
                   if isinstance(param, LeakyReLU):
                       param.load_state_dict()
                   if isinstance(param, ELU):
                       param.load_state_dict()
           self.set_cpu()
