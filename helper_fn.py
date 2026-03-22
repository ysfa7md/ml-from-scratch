import numpy as np

class HelpFn:
    def __init__(self):
        pass

    @staticmethod
    def logit(p):
        '''logit function
        Args:
            p (float): range [0,1]
        Returns:
            logit(float): The return value. logit of p. in range [-inf, inf]
        '''
        return np.log(p/(1-p))

    @staticmethod
    def sigmoid(z):
        '''sigmoid function (logit Inverse)
        Args:
            z (float): range [-inf, inf]
        Returns:
            sigmoid(float): The return value. sigmoid of z. in range [0,1]
        '''
        return 1/(1+np.exp(-z))

