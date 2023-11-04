import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# https://towardsdatascience.com/building-a-convolutional-neural-network-from-scratch-using-numpy-a22808a00a40
# https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939

class ConvolutionLayer:
    '''
    Convolution layer for a Convolutional Neural Network

    Attributes
    ----------
    `kernel_num: int`
        The number of kernels (filters) associated with the convolution layer
    
    `kernel_size: int`
        The size of the kernels (filters) associated with the convolution layer

    Methods
    -------
    `segmentGenerator(image: np.array)` -> `height`, `width`
        Generate smaller segments of an image
    
    `forwardProp(image: np.array)` -> `convolution_output: np.array`
        Carries out the convolution for each generated portion of the image

    `back_prop(dE_dY, alpha)` -> `dE_dK: np.array`
        Calculate the gradient of the loss function
    '''

    def __init__(self, kernel_num: int, kernel_size: int) -> None:
        '''
        Initialise the Convolution layer of a Convolutional Neural Network

        Parameters
        ------
        `kernel_num: Integer`
            The number of kernels (filters) to pass through
        
        `kernel_size: Integer`
            The size of the kernels (filters)

        Returns
        -------
        `None`

        Example
        -------
        `cl = ConvolutionalLayer(2, 3)`
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
        `image: np.array`
            numpy array representation of the image

        Returns
        -------
        `Portion: np.array`
            The portion of the image
        
        `Height: int`
            The starting height value of the portion
        
        `Width: int`
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
        `image: np.array`
            numpy array representation of the image
        
        Returns
        -------
        `convolution_output: np.array`
            Output of the convolution layer
        '''
        
        image_height,image_width,*_ = image.shape

        # Create an array of zeros with the size of the convolution output that will be modified.
        convolution_output = np.zeros((image_height-self.kernel_size+1, image_width-self.kernel_size+1, self.kernel_num))

        # Run convolution on each segment of the image
        # Convolution = Sum of Segments x Kernels to apply bias
        for segment, h, w in self.segmentGenerator(image):
                    
            convolution_output[h,w] = np.sum(segment*self.kernels)
        
        return convolution_output
    
    def back_prop(self, dE_dY, alpha):
        '''
        Calculate the gradient of the loss function

        Parameters
        ----------
        `dE_dY`
            Derivative of the error in respect to the output
        
        `alpha`
            The learning rate of the model

        Returns
        -------
        `dE_dK: np.array`
            Derivative of the error in respect to the kernels
        '''
        dE_dk = np.zeros(self.kernel.shape)
        for patch, h, w in self.segmentGenerator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        self.kernel -= alpha*dE_dk
        return dE_dk

class PoolingLayer:

    def __init__(self, kernel_size: int) -> None:
        '''
        Initialise the Pooling layer of a Convolutional Neural Network

        Parameters
        ----------
        `kernel_size: Integer`
            The size of the kernels (filters)
        
        Returns
        -------
        `None`

        Example
        -------
        `pl = PoolingLayer(2)`
        '''
        self.kernel_size = kernel_size

    def segmentGenerator(self, image):
        '''
        Generate smaller segments of an image

        Parameters
        ----------
        `image: np.array`
            numpy array representation of the image

        Returns
        -------
        `Portion: np.array`
            The portion of the image
        
        `Height: int`
            The starting height value of the portion
        
        `Width: int`
            The starting width value of the portion

        ### Yields each portion of the image with respect to the kernel size
        '''

        # Height of each segment in reference to kernels
        height = image.shape[0] // self.kernel_size
        width = image.shape[1] // self.kernel_size
        self.image = image

        for h in range(height):
            for w in range(width):
                segment = image[(h*self.kernel_size):(h*self.kernel_size+self.kernel_size), (w*self.kernel_size):(w*self.kernel_size+self.kernel_size)]
                yield segment, h, w

    def forward_prop(self, image):
        '''
        Carries out the max pooling for each generated portion of the image

        Parameters
        ----------
        image: np.array
            numpy array representation of the image

        Returns
        -------
        max_pooling_output: np.array
            Output of the max pooling layer
        '''
        image_h, image_w, num_kernels = image.shape
        # Max val in each segment
        max_pooling_output = np.zeros((image_h//self.kernel_size, image_w//self.kernel_size, num_kernels))
        for segment, h, w in self.segmentGenerator(image):
            # np.amax -> Array maximum value of each segment
            max_pooling_output[h,w] = np.amax(segment, axis=(0,1))
        return max_pooling_output
    
    def back_prop(self, dE_dY):
        '''
        Calculate the gradient of the loss function

        Parameters
        ----------
        `dE_dY`
            Derivative of the error in respect to the output
        
        Returns
        -------
        `dE_dK: np.array`
            Derivative of the error in respect to the kernels
        '''

        # Initialise the array for holding error
        dE_dk = np.zeros(self.image.shape)
        for patch,h,w in self.patches_generator(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0,1))
            
            for height_index in range(image_h):
      
                for width_index in range(image_w):
                    
                    for kernel_index in range(num_kernels):

                        if patch[height_index,width_index,kernel_index] == max_val[kernel_index]:
      
                            dE_dk[h*self.kernel_size+height_index, w*self.kernel_size+width_index, kernel_index] = dE_dY[h,w,kernel_index]
            return dE_dk

class SoftmaxLayer:
    
    def __init__(self, input_units, output_units):
        self.weight = np.random.randn(input_units, output_units)/input_units
        self.bias = np.zeros(output_units)

    def forward_prop(self, image):
        self.original_shape = image.shape
        image_flattened = image.flatten()
        self.flattened_input = image_flattened
        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output
        softmax_output = np.exp(first_output) / np.sum(np.exp(first_output), axis=0)
        return softmax_output

    def back_prop(self, dE_dY, alpha):
        for i, gradient in enumerate(dE_dY):
            if gradient == 0:
                continue
            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
            dY_dZ[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)

            dZ_dw = self.flattened_input
            dZ_db = 1
            dZ_dX = self.weight

            dE_dZ = gradient * dY_dZ

            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ

            self.weight -= alpha*dE_dw
            self.bias -= alpha*dE_db

            return dE_dX.reshape(self.original_shape)

class FullyConnectedLayer:

    def __init__(self) -> None:
        pass

class CNN:

    def __init__(self) -> None:
        pass

if __name__ == "__main__":

    cl = ConvolutionLayer(5, 3)
    filename = "image.jpg"
    image = Image.open(filename).convert("L")
    image.save(f"{filename}_greyscale.jpg")
    image = np.asarray(image)
    img = cl.forwardProp(image)

    for i in range(len(img[0][0])):
        plt.imshow(img[:,:,i], cmap='gray')
        plt.savefig(f"{filename}_output_{i}.jpg", dpi=img.shape[0])
        plt.show()