import numpy as np

class ConvolutionLayer:
    '''
    Convolution layer for a Convolutional Neural Network

    Attributes
    ----------
    kernel_num: int
        The number of kernels (filters) associated with the convolution layer
    
    kernel_size: int
        The size of the kernels (filters) associated with the convolution layer

    Methods
    -------
    portionGenerator(image: np.array) -> height, width
    '''

    def __init__(self, kernel_num: int, kernel_size: int) -> None:
        '''
        Initialise the Convolution layer of a Convolutional Neural Network

        Parameters
        ------
        kernel_num: Integer
            The number of kernels (filters) to pass through
        
        kernel_size: Integer
            The size of the kernels (filters)

        Returns
        -------
        None

        Example
        -------
        cl = ConvolutionalLayer(2, 3)
        '''
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        # Generate kernels x*y in size with random variables to be trained
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / kernel_size ** 2

    def portionGenerator(self, image: np.array):
        '''
        
        '''
        pass

class PoolingLayer:

    def __init__(self) -> None:
        pass

class FullyConnectedLayer:

    def __init__(self) -> None:
        pass

class CNN:

    def __init__(self) -> None:
        pass