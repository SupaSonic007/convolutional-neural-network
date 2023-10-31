import numpy as np
from PIL import Image
from matplotlib import cm

# https://towardsdatascience.com/building-a-convolutional-neural-network-from-scratch-using-numpy-a22808a00a40
# https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939

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
    segmentGenerator(image: np.array) -> height, width
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

        # Generate kernels x*y in size with random variables to be trained, normalisation by with kernel_size ** 2
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / kernel_size ** 2

    def segmentGenerator(self, image: np.array):
        '''
        Generate smaller segments of an image

        Parameters
        ----------
        image: np.array
            numpy array representation of the image

        Returns
        -------
        Portion: np.array
            The portion of the image
        
        Height: int
            The starting height value of the portion
        
        Width: int
            The starting width value of the portion

        ### Yields each portion of the image with respect to the kernel size
        '''
        height, width, *_ = image.shape
        self.image = image

        for h in range(height-self.kernel_size+1):
            for w in range(width-self.kernel_size+1):
                segment = image[h:h+(self.kernel_size), w:w+(self.kernel_size)]
                yield segment, h, w

    def forwardProp(self, image: np.array):
        '''
        Carries out the convolution for each generated portion of the image

        Parameters
        ----------
        image: np.array
            numpy array representation of the image
        
        Returns
        -------
        convolution_output: np.array
            Output of the convolution layer
        '''
        
        image_height,image_width,*_ = image.shape

        # Create an array of zeros with the size of the convolution output that will be modified.
        convolution_output = np.zeros((image_height-self.kernel_size+1, image_width-self.kernel_size+1, self.kernel_num))

        # Run convolution on each segment of the image
        # Convolution = Sum of Segments x Kernels to apply bias
        for segment, h, w in self.segmentGenerator(image):
            convolution_output[h,w] = np.sum(segment*self.kernels, axis=(1,2))
        
        return convolution_output
    
    def back_prop(self, dE_dY, alpha):
        '''
        Calculate the gradient of the loss function

        Parameters
        ----------
        dE_dY
            Derivative of the error in respect to the output
        
        alpha
            The learning rate of the model

        Returns
        -------
        dE_dK: np.array
            Derivative of the error in respect to the kernels
        '''
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        self.kernels -= alpha*dE_dk
        return dE_dk

class PoolingLayer:

    def __init__(self) -> None:
        pass

class FullyConnectedLayer:

    def __init__(self) -> None:
        pass

class CNN:

    def __init__(self) -> None:
        pass

if __name__ == "__main__":

    cl = ConvolutionLayer(1, 3)
    
    image = Image.open('image.jpg').convert("L")
    image.save("L.jpg")
    image = np.asarray(image)
    # Allow any size of pixels for saving
    Image.MAX_IMAGE_PIXELS = None
    img = cl.forwardProp(image)

    newimg = None
    percent = 0.0
    # Concatenate all np arrays to put together
    # for i in range(len(img)):
    #     if i == 0: newimg = img[0]; continue
    #     newimg = np.concatenate((newimg, img[i]), axis=1)
    #     if percent != np.floor(i/len(img)*100):
    #         percent = np.floor(i/len(img)*100)
    #         print(f"{percent}%")
    print("100.0%")
    Image.MAX_IMAGE_PIXELS = None
    print(len(img), len(img[0]), len(img[0][0]))
    print(img[0][0])
    Image.fromarray(img, mode="L").convert('RGB').save("coolio.jpg")