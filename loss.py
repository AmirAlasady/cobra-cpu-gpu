import cupy as cp # type: ignore
import numpy as np


class LossBase:
    # Epsilon to avoid numerical instabilities
    epsilon = 1e-12

class Loss_on_cpu(LossBase):
    @staticmethod
    def mse(y_true, y_pred):
        """Mean squared error loss."""
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def mse_prime(y_true, y_pred):
        # You can use y_true.shape[1] if y_true is 2D (batch_size) instead of np.size.
        return 2 * (y_pred - y_true) / np.size(y_true)
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        """
        Binary cross-entropy loss.
        (You may call this binary_cross_entropy if preferred.)
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, LossBase.epsilon, 1 - LossBase.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def cross_entropy_prime(y_true, y_pred):
        """
        Derivative of binary cross-entropy loss.
        Note: Some frameworks use (y_pred - y_true)/batch_size.
        """
        y_pred = np.clip(y_pred, LossBase.epsilon, 1 - LossBase.epsilon)
        return (y_pred - y_true) / np.size(y_true)
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """
        Categorical cross-entropy loss.
        Suitable when using softmax outputs.
        """
        y_pred = np.clip(y_pred, LossBase.epsilon, 1 - LossBase.epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    @staticmethod
    def categorical_cross_entropy_prime(y_true, y_pred):
        """
        Derivative for categorical cross-entropy loss.
        Often, when combined with softmax, the derivative simplifies to (y_pred - y_true).
        """
        return y_pred - y_true

    # If you prefer the name "binary_cross_entropy", you can simply alias it:
    binary_cross_entropy = cross_entropy
    binary_cross_entropy_prime = cross_entropy_prime


class Loss_on_gpu(LossBase):
    @staticmethod
    def mse(y_true, y_pred):
        """Mean squared error loss on GPU."""
        return cp.mean(cp.power(y_true - y_pred, 2))
    
    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / cp.size(y_true)
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        """
        Binary cross-entropy loss on GPU.
        (Assumes y_true and y_pred are already cupy arrays.)
        """
        y_pred = cp.clip(y_pred, LossBase.epsilon, 1 - LossBase.epsilon)
        return cp.mean(-y_true * cp.log(y_pred) - (1 - y_true) * cp.log(1 - y_pred))
    
    @staticmethod
    def cross_entropy_prime(y_true, y_pred):
        y_pred = cp.clip(y_pred, LossBase.epsilon, 1 - LossBase.epsilon)
        return (y_pred - y_true) / cp.size(y_true)
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        y_pred = cp.clip(y_pred, LossBase.epsilon, 1 - LossBase.epsilon)
        return -cp.mean(y_true * cp.log(y_pred))
    
    @staticmethod
    def categorical_cross_entropy_prime(y_true, y_pred):
        return y_pred - y_true
    def accuracy(y_true, y_pred):
        predictions = cp.argmax(y_pred, axis=0)
        labels = cp.argmax(y_true, axis=0)
        return cp.mean(predictions == labels)
    # Optionally alias these names to match the CPU version
    binary_cross_entropy = cross_entropy
    binary_cross_entropy_prime = cross_entropy_prime
