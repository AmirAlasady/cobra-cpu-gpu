import cupy as cp # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from dense_layer import Dense  # Assumes your Layer classes define set_gpu, set_cpu, forward, backward, state_dict, load_state_dict

class Sequential:
    def __init__(self, network: list, device: str = 'cpu'):
        """
        Initialize with a list of layers and set the device ('cpu' or 'gpu').
        """
        self.layers = network
        assert device in ['cpu', 'gpu'], "Device must be either 'cpu' or 'gpu'."
        self.device = device

        # Automatically set each layer to the proper device.
        if self.device == 'gpu':
            for layer in self.layers:
                layer.set_gpu()
        else:
            for layer in self.layers:
                layer.set_cpu()

    def set_gpu(self):
        """Move all layers to the GPU."""
        if self.device == 'gpu':
            print('Sequential is already on gpu')
        else:
            self.device = 'gpu'
            for layer in self.layers:
                layer.set_gpu()
            print('Moved to gpu!')

    def set_cpu(self):
        """Move all layers to the CPU."""
        if self.device == 'cpu':
            print('Sequential is already on cpu')
        else:
            self.device = 'cpu'
            for layer in self.layers:
                layer.set_cpu()
            print('Moved to cpu!')

    def forward(self, input_):
        """
        Forward propagation through all layers.
        Each layer's output becomes the next layer's input.
        """
        output = input_
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad):
        """
        Backward propagation: iterate in reverse through layers,
        passing the gradient and learning rate.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self, epochs, X, Y, loss_fn, loss_prime, optimizer=None, verbose=True, plot=False):
        """
        Train the network:
          - X and Y are lists (or iterables) of training samples and corresponding labels.
          - loss and loss_prime are functions for loss calculation and its derivative.
          - lr is the learning rate.
        """
        self.loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X, Y):
                # Forward pass (assumes x and y are already on the correct device if using GPU)
                output = self.forward(x)
                total_loss += loss_fn(y, output)
                # Compute the gradient from the loss and backpropagate it
                grad = loss_prime(y, output)
                self.backward(grad)
                # optimizer stepping
                optimizer.step(self.layers)
            # Average loss over the number of samples
            avg_loss = total_loss / len(X)
            self.loss_history.append(avg_loss)
            print(f"{epoch + 1}/{epochs}, error={avg_loss}")

        # Plot loss history
        if plot:
            # If on GPU, convert each loss value to a CPU scalar if necessary.
            if self.device == 'gpu':
                loss_history_cpu = [
                    loss_val.item() if hasattr(loss_val, 'item') else loss_val
                    for loss_val in self.loss_history
                ]
                plt.plot(loss_history_cpu)
            else:
                plt.plot(self.loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss During Training')
            plt.show()

    def state_dict(self):
        model_state_dict = {}
        for i, layer in enumerate(self.layers):
            layer_id = layer.id  # Use the unique layer ID for keys
            model_state_dict[layer_id] = layer.state_dict()
        # Append the "device" key-value pair at the end
        model_state_dict["device"] = self.device

        return model_state_dict

    def load_state_dict(self, state_dict):
        
        # set state dict overwrite device attribute 
        if state_dict['device']=='gpu':
           keys = state_dict.keys()
           keys=list(keys)
           for i, layer in enumerate(self.layers):
               if isinstance(layer, Dense):
                   layer_id = layer.id
                   key=keys[i]
                   layer.load_state_dict(state_dict[key])
               else:
                   pass
           self.set_gpu()
        else:
           keys = state_dict.keys()
           keys=list(keys)
           for i, layer in enumerate(self.layers):
               if isinstance(layer, Dense):
                   layer_id = layer.id
                   key=keys[i]
                   layer.load_state_dict(state_dict[key])
               else:
                   pass
           self.set_cpu()