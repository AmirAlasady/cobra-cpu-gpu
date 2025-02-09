## Cobra Documentation:

**What is Cobra?**

Cobra is a powerful Python library designed to simplify the creation, development, and training of Artificial Intelligence (AI) and Machine Learning (ML) models. It caters to both neural networks and customizable models, empowering you to build and deploy your AI solutions efficiently.

**How Does It Work?**

Cobra operates on the fundamental concept of a single neuron represented as a layer. Each layer is initialized with random weights and biases based on the specified architecture. Forward propagation (FP) and backward propagation (BP) calculations are then performed within the layer. This concept can be extended by adding numerous neurons and layers in a sequential order, with each layer independently executing FP and BP.

**Cobra Features:**

* **Versatility:** Craft neural networks in various configurations, including single-layer, multilayer, and sequential models.
* **Performance:** Leverage the power of CPUs or harness the speed of GPUs for faster training and improved performance, especially with larger models.
* **Automation:** Automate the entire forward/backward propagation and training processes with just a single function call, streamlining your workflow.
* **Activation Functions:** Utilize a variety of built-in activation functions like Tanh, Sigmoid, ReLU, and Softmax to introduce non-linearity into your models.
* **Loss Methods and Optimizers:** Employ various loss functions with their corresponding derivatives and choose from different optimizers to fine-tune your model's learning process.
* **Customization:** Gain fine-grained control over your model's architecture, activation functions, and training process through a dedicated custom builder.
* **Deep Customization:** For ultimate control, override the base abstract methods to customize the entire model building process, including architecture, activation functions, forward/backward propagation, and training.
* **Transfer Learning:** Facilitate transfer learning by extracting model parameters through the state dictionary API, saving them as `.pkl` files, and loading them into other models without retraining, saving time and resources.

**Cobra Workflow:**

1. **Data Preparation:** Meticulously prepare your dataset for training, ensuring it is clean, well-structured, and suitable for your chosen task.
2. **Model Creation:** Select or create a neural network or custom model that aligns with your specific requirements and data characteristics.
3. **Model Training:** Train the model on your prepared dataset, iteratively adjusting its parameters to achieve optimal performance.
4. **Model Usage:** Utilize the trained model to make predictions or inferences on new data, gaining valuable insights or accomplishing your desired tasks.

**Main Documentation:**

**0. Importing the Library:**

```python
from cobra import *
```

**1. Layers:**

**a) Dense:**

* `Layer(n, m)`: Creates a layer with `n` input neurons and `m` output neurons.

**b) Activation:**

* `Tanh()`, `Sigmoid()`, `ReLU()`, and `Softmax()`: Create instances of the corresponding activation functions.

**2. Models:**

**a) Separated Layers:**

```python
l1 = Layer(n, m)
activation1 = Tanh()
l2 = Layer(m, v)
# ... (add more layers and activations)
```

**b) Sequential Object:**

```python
network = Sequential([
    Layer(x, m),
    Tanh(),
    Layer(m, v),
    Sigmoid(),
    # ... (add more layers and activations)
])
```

**c) Custom Model:**

```python
class Model_1(Model):
    def __init__(self):
        self.l1 = Layer(2, 20)
        self.l2 = Sigmoid()
        self.network1 = Sequential([
            Layer(20, 9999),
            Tanh(),
            Layer(9999, 345),
            Tanh(),
            # ... (add more layers)
        ])
        self.network2 = Sequential([
            Layer(345, 9999),
            Tanh(),
            Layer(9999, 1),
            Tanh(),
            # ... (add more layers)
        ])
        self.device='cpu (default)' <or> 'gpu'
        super().set()
model1 = Model_1()
```





## Device Configuration for Cobra:

**3. Device Configuration:**

**a) Using the `device` Attribute:**

- Directly specify the desired device when creating layers, activations, or models:

```python
# CPU

l1 = Layer(n, m, device='cpu')  # Default: device='cpu'
activation1 = Tanh(device='cpu')

network_cpu = Sequential([
    Layer(x, m, device='cpu'),
    Tanh(device='cpu'),
    Layer(m, v, device='cpu'),
    Sigmoid(device='cpu'),
], device='cpu')  # Set entire network to CPU

network_cpu = Sequential([
    Layer(x, m, device='gpu'),
    Tanh(device='gpu'),
    Layer(m, v, device='gpu'),
    Sigmoid(device='gpu'),
], device='gpu')  # Set entire network to GPU

class Model_1(Model):               
     def __init__(self):
         ....
         ....
         self.device='cpu (default)' <or> 'gpu'
         super().set()
```

**b) Using the `set_` Method:**

- Set the device after creating layers, activations, or models:

```python
l1 = Layer(n, m)
l1.set_cpu()  # Set to CPU
l1.set_gpu() # Set to GPU

activation1 = Tanh()
activation1.set_cpu()  # Set CPU 
activation1.set_gpu()  # Set GPU

network = Sequential([
    Layer(x, m),
    Tanh(),
    Layer(m, v),
    Sigmoid(),
])
network.set_cpu()  # Set entire network to CPU
network.set_gpu()  # Set entire network to GPU

class Model_1(Model):               
     def __init__(self):
         ....
         ....
         self.device='cpu (default)' <or> 'gpu'
         super().set()
model1 = Model_1()
model1.set_cpu()  # Set entire custom model to CPU
model1.set_gpu()  # Set entire custom model to GPU
```

**Important:**

- Ensure compatible graphics drivers for GPU usage.
- Consult your hardware documentation for specific GPU capabilities and limitations.




## Cobra Forward Propagation, Loss & Optimization, Backpropagation:

**4. Forward Propagation:**

**a) Layers:**

```python
l1 = Layer(c, v)
output = l1.forward(input_data)  # Perform forward pass and store output
```

**b) Activation Functions:**

```python
l2 = Tanh()
output = l2.forward(input_data)  # Apply activation and store output
```

**c) Sequential Objects:**

```python
network = Sequential([
    Layer(c, v),
    Tanh(),
    Layer(v, n),
])
output = network.forward(input_data)  # Forward pass through all layers
```

**d) Custom Models:**

```python
model1 = Model_1()
output = model1.forward(input_data)  # Forward pass through custom model
```

**5. Loss and Optimization Functions:**

Cobra offers both CPU and GPU-based loss functions with corresponding derivatives:

**a) CPU-based Loss (Loss_on_cpu):**

- `Loss_on_cpu.mse(y_true, y_pred)` (mean squared error) with `Loss_on_cpu.mse_prime(y_true, y_pred)` derivative
- `Loss_on_cpu.binary_cross_entropy(y_true, y_pred)` with `Loss_on_cpu.binary_cross_entropy_prime(y_true, y_pred)` derivative

**b) GPU-based Loss (Loss_on_gpu):**

- `Loss_on_gpu.mse(y_true, y_pred)` (mean squared error) with `Loss_on_gpu.mse_prime(y_true, y_pred)` derivative
- `Loss_on_gpu.binary_cross_entropy(y_true, y_pred)` with `Loss_on_gpu.binary_cross_entropy_prime(y_true, y_pred)` derivative

**6. Backpropagation:**

**a) Layers:**

```python
l1.backward(grad, lr)  # Propagate the gradient back through the layer with learning rate
```

**b) Activation Functions:**

```python
l2.backward(grad, lr)  # Propagate gradient back through the activation function
```

**c) Sequential Objects:**

```python
network.backward(grad, lr)  # Backpropagate through all layers in the sequence
```

**d) Custom Models:**

```python
model1.backward(grad, lr)  # Backpropagate through the custom model's layers
```
**KEEP IN MIND THAT FOR FULL CUSTOMISABLE CUSTOM MODEL YOU WILL BE ABLE TO OVERWRITE THE HALL FORWARD AND BACKWARD AND TRAIN METHODS OF THE BASE MODEL **
```python

class Q(Model):
    def __init__(self):
        self.l1 = Layer(2, 2)
        self.l4 = Sequential([
            Layer(2, 3),
            ReLU(),
            Layer(3, 1),
            Tanh(),
        ])
        self.device = 'cpu'
        super().set()

    def forward(self, x):
        # Assuming x is a 2D tensor of shape (batch_size, 2)
        out = self.l1(x)  # Pass x through the first layer
        out = self.l4(out)  # Pass output through the sequential layers
        return out  # Return the final output

```
**Additional Notes:**

- Forward propagation calculates outputs based on inputs and layers/activations.
- Loss functions measure the difference between predicted and actual output, returning a scalar value.
- Optimizers use the loss and learning rate to update weights and biases in the model.
- Backpropagation propagates the loss gradient through the network to adjust weights and biases.

## Cobra Training, Saving & Loading:

**7. Training Method:**

**a) Sequential and Custom Models:**

- These objects already have a default `train` method:

```python
def train(self, epochs, X, Y, lr, loss, loss_prime, plot=True):
    # ... training logic ...
```

- Call it like this:

```python
network = Sequential([Layer(c, v), Layer(v, n)])
network.train(4000, X, Y, 0.01, Loss_on_cpu.mse, Loss_on_cpu.mse_prime)

model1 = Model_1()
model1.train(4000, X, Y, 0.01, Loss_on_cpu.mse, Loss_on_cpu.mse_prime)
```

**b) Manual Training for Separated Layers:**

```python
def train(m,e,X,Y,lr):
                # m = model network layers as list
                # e = number of epochs
                # X = inputs 
                # Y = True labels
                # lr = learning rate 
                loss_history=[]
                for i in range(e):
                    error=0
                    for x,y in zip(X,Y):
                        output=m.forward(x)
                        
                        error += Loss_on_cpu.mse(y,output)
                        
                        grad=Loss_on_cpu.mse_prime(y,output)
                        
                        grad=m.backward(grad,lr)
                    error /= len(x)
                    loss_history.append(error)
                    if True:
                        print(f"{i + 1}/{e}, error={error}")
    
```

**8. Saving and Loading Models:**

**a) Saving Model State:**

1. Access model parameters with `state_dict()`:

```python
l1 = Layer(c, v)
parameters = l1.state_dict()  # Access parameters of a layer

network = Sequential([Layer(c, v), Layer(v, n)])
parameters = network.state_dict()  # Access parameters of a Sequential object

model1 = Model_1()
parameters = model1.state_dict()  # Access parameters of a custom model
```

2. Save to a `.pkl` file:

```python
save_model_parameters(model, 'my_model.pkl')  # Automatic state_dict access and saving
```

**b) Loading Model State:**

1. Load the `.pkl` file:

```python
trained_model = load_state_dict_from_file('my_model.pkl')
```

2. Build the same model architecture:

```python
# For Sequential object
model1 = Sequential([Layer(c, v), Layer(v, n)])

# For custom model
model1 = Model_1()
```

3. Load state dictionary into the new model:

```python
model1.load_state_dict(trained_model)
```

**Notes:**

- Loading automatically sets the device of the new model to the loaded model's device.
- Both `save_model_parameters` and `load_state_dict_from_file` handle parameter transfer automatically.

This explanation clarifies Cobra's training, saving, and loading functionalities, empowering you to effectively manage your AI models!





 **Here's a comprehensive breakdown of the provided Cobra code example, designed to run on the CPU:**

**1. Importing Necessary Modules:**

```python
from cobra import *
```
- Imports the core Cobra library for neural network functionality.


**2. Defining Inputs and Outputs:**

```python
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))  # XOR gate inputs
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))               # XOR gate true outputs
```

**3. Creating a Custom Model:**

```python
class model1(Model):
    def __init__(self):
        self.l1 = Layer(2, 2)
        self.l2 = Sequential([
            Layer(2, 3),
            ReLU(),
            Layer(3, 1),
            Sigmoid()
        ])
        self.device = 'cpu'  # Force model to run on CPU
        super().set()  # Initialize model components
```

- Defines a custom model class `model1` that inherits from the base `Model` class.
- Contains two layers: `l1` (Layer object) and `l2` (Sequential object with multiple layers).
- Explicitly sets the device to 'cpu' for CPU-based execution.

**4. Creating and Training the Model:**

```python
model = model1()  # Instantiate the model
model.train(15000, X, Y, 0.01, Loss_on_cpu.mse, Loss_on_cpu.mse_prime)  # Train the model
```
1/15000, error=0.701
.....

14859/15000, error=0.0012815110752247886


14860/15000, error=0.0012813934669477795


14861/15000, error=0.001281275887767244


![dbai](https://github.com/AmirAlasady/cobra-cpu-gpu/assets/96434221/72fdb305-b759-43e8-8a46-b7043d1dd3cb)

**5. Visualizing Decision Boundaries:**

```python
# Code for generating a 3D plot of decision boundaries
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        #x=cp.asarray(x)
        #y=cp.asarray(y)
        z= model.forward([[x], [y]])
        points.append([x, y, z[0,0]])
points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
```
![function_shape](https://github.com/AmirAlasady/cobra-cpu-gpu/assets/96434221/a6a6ba56-7821-41cf-bb22-10f3e9ee171b)

**6. Testing the Model:**

```python
# Code for testing the model with sample inputs
print(f'__|__|')
print(f'00|0 | {model.forward([[0],[0]])}')
print(f'01|1 | {model.forward([[0],[1]])}')
print(f'10|1 | {model.forward([[1],[0]])}')
print(f'11|0 | {model.forward([[1],[1]])}')
```
00|0 | [[0.01050455]]

01|1 | [[0.96597505]]

10|1 | [[0.96596509]]

11|0 | [[0.01041758]]

**7. Saving the Trained Model:**

```python
save_model_parameters(model, 'my_Ai.pkl')  # Save model parameters to a file
```
<===== Saving =====>
<===== Done =====>
**Key Points:**

- The code demonstrates a custom neural network trained to learn the XOR gate using Cobra's CPU-based functionality.
- It highlights model creation, training, testing, visualization, and saving capabilities.
- The `device='cpu'` setting ensures the model runs on the CPU.



Here's the corrected and improved code example for running the XOR gate network on the GPU in Cobra:

```python
from cobra import *
import numpy as np
import cupy as cp  # Import cupy for GPU operations

# Inputs and outputs (move directly to GPU memory)
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
X = cp.asarray(X)  # Move X to GPU
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
Y = cp.asarray(Y)  # Move Y to GPU

# Custom model (no Layer(2, 2) needed)
class model1(Model):
    def __init__(self):
        self.l2 = Sequential([
            Layer(2, 3),
            ReLU(),
            Layer(3, 1),
            Sigmoid(),
        ])
        self.set_gpu()  # Set the model to use the GPU

# Create and train the model on GPU
model = model1()
model.train(12000, X, Y, 0.01, Loss_on_gpu.mse, Loss_on_gpu.mse_prime)

# ... (rest of your code: visualization, testing, saving)
```

**Improvements:**

- **GPU data transfer:** Data (`X` and `Y`) is directly moved to GPU memory using `cp.asarray` for efficiency.
- **Model device setting:** The `set_gpu()` method is used explicitly within the `model1` class to ensure the model resides on the GPU.
- **Layer clarification:** The unnecessary `Layer(2, 2)` has been removed.

Remember to ensure your system has NVIDIA GPUs with compatible drivers for GPU utilization. This revised code ensures your Cobra network leverages the GPU's power for faster training.


## Heavy Load Test for GPU/CPU Performance

**Important Note:** Your GPU's speed significantly impacts performance. Although GPUs excel for deep networks, CPUs might be faster for smaller ones.

**Benchmarking on CPU:**

```python
import time

# Heavy model with large hidden layers
class model1(Model):
    def __init__(self):
        self.l2 = Sequential([
            Layer(2, 9999),
            ReLU(),
            Layer(9999, 9999),
            Sigmoid(),
        ])
        self.l3 = Sequential([
            Layer(9999, 9999),
            ReLU(),
            Layer(9999, 9999),
            Sigmoid(),
        ])
        self.device = 'cpu'  # Set to CPU for this test
        super().set()

# Create model and run 3 training rounds
model = model1()
start_time = time.time()
model.train(3, X, Y, 0.01, Loss_on_cpu.mse, Loss_on_cpu.mse_prime)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time on CPU (3 rounds): {elapsed_time:.2f} seconds")
```
Elapsed time on CPU: 27.76002335548401 seconds

now on GPU


**Code:**

```python
import time

# Define a model with large hidden layers
class model1(Model):
    def __init__(self):
        self.l2 = Sequential([
            Layer(2, 9999),
            ReLU(),
            Layer(9999, 9999),
            Sigmoid(),
        ])
        self.l3 = Sequential([
            Layer(9999, 9999),
            ReLU(),
            Layer(9999, 9999),
            Sigmoid(),
        ])
        self.device = 'gpu'  # Explicitly set device to GPU
        super().set()

# Create the model and measure training time on GPU
model = model1()
start_time = time.time()
model.train(3, X, Y, 0.01, Loss_on_gpu.mse, Loss_on_gpu.mse_prime)  # Use GPU-specific loss functions
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time on GPU (3 rounds): {elapsed_time:.2f} seconds")
```
Elapsed time on GPU: 5.094772577285767 seconds

thus the finall results are 

for CPU: 27.76002335548401 seconds

for GPU: 5.094772577285767 seconds
