
import cupy as cp
import numpy as np

class loss_base:
    pass

class Loss_on_cpu(loss_base):
    
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


# updated to 2.1 version avoiding auto data movment to gpu and relaying user conversion
class Loss_on_gpu(loss_base):
    
    @staticmethod
    def mse(y_true, y_pred):
        #yt=cp.asarray(y_true)
        #yp=cp.asarray(y_pred)
        return cp.mean(cp.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        #yt=cp.asarray(y_true)
        #yp=cp.asarray(y_pred)
        return 2 * (y_pred - y_true) / cp.size(y_true)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        #yt=cp.asarray(y_true)
        #yp=cp.asarray(y_pred)
        return cp.mean(-y_true * cp.log(y_pred) - (1 -y_true) * cp.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        #yt=cp.asarray(y_true)
        #yp=cp.asarray(y_pred)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / cp.size(y_true)
    
