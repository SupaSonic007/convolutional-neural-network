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
    `segmentGenerator(image: np.array)` -> `height, width`
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
    '''
    Pooling layer for a Convolutional Neural Network

    Attributes
    ----------
    `kernel_size: int`
        The size of the kernels (filters) associated with the pooling layer

    Methods
    -------
    `segmentGenerator(image: np.array)` -> `segment, height, width`
        Generate smaller segments of an image
    
    `forwardProp(image: np.array)` -> `max_pooling_output: np.array`
        Carries out the max pooling for each generated portion of the image

    `back_prop(dE_dY, alpha)` -> `dE_dK: np.array`
        Calculate the gradient of the loss function
    '''

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

        # Initialise the array for holding the error
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

class FullyConnectedLayer:
    '''
    Fully connected layer for a Convolutional Neural Network

    Attributes
    ----------
    `input_units: int`
        The number of input units of the fully connected layer

    `output_units: int`
        The number of output units of the fully connected layer

    Methods
    -------
    `forward_prop(image: np.array)` -> `softmax_output: np.array`
        Carries out the softmax for each generated portion of the image

    `back_prop(dE_dY, alpha)` -> `dE_dX: np.array`
        Calculate the gradient of the loss function
    '''
    
    def __init__(self, input_units, output_units):
        '''
        Initialise the Fully connected layer of a Convolutional Neural Network

        Parameters
        ----------
        `input_units: int`
            The number of input units of the fully connected layer

        `output_units: int`
            The number of output units of the fully connected layer
        
        Returns
        -------
        `None`

        Example
        -------
        `fcl = FullyConnectedLayer(2, 3)`
        '''
        self.input_units = input_units
        self.output_units = output_units
        self.weight = np.random.randn(input_units, output_units)/input_units
        self.bias = np.zeros(output_units)

    def forward_prop(self, image):
        '''
        Carries out the sigmoid on the image

        Parameters
        ----------
        `image: np.array`
            numpy array representation of the image

        Returns
        -------
        sigmoid_output: float
            Output of the sigmoid function
        '''
        self.original_shape = image.shape
        image_flattened = image.flatten()
        self.flattened_input = image_flattened
        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output
        sigmoid_output = np.exp(first_output) / np.sum(np.exp(first_output), axis=0)
        return sigmoid_output

    def back_prop(self, dE_dY, alpha):
            '''
            Calculate the gradient of the loss function

            Parameters
            ----------
            `dE_dY`: np.array
                Derivative of the error in respect to the output
            
            `alpha`: float
                The learning rate of the model

            Returns
            -------
            `dE_dX: np.array`
                Derivative of the error in respect to the kernels
            '''

            # Loop through each gradient in the derivative of the error in respect to the output
            for i, gradient in enumerate(dE_dY):
                if gradient == 0:
                    continue

                # Calculate the transformation equation and total sum
                transformation_eq = np.exp(self.output)
                S_total = np.sum(transformation_eq)

                # Calculate dY_dZ (derivative of the output in respect to the input)
                dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
                dY_dZ[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)

                # Calculate dZ_dw, dZ_db, and dZ_dX
                dZ_dw = self.flattened_input
                dZ_db = 1
                dZ_dX = self.weight

                # Calculate dE_dZ
                dE_dZ = gradient * dY_dZ

                # Calculate dE_dw, dE_db, and dE_dX
                dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
                dE_db = dE_dZ * dZ_db
                dE_dX = dZ_dX @ dE_dZ

                # Update weight and bias using gradient descent
                self.weight -= alpha*dE_dw
                self.bias -= alpha*dE_db

                # Reshape dE_dX to its original shape
                return dE_dX.reshape(self.original_shape)

def CNN_forward(image, label, layers):
    '''
    Forward propagation of a convolutional neural network.
    [Runs through the network]

    Parameters
    ----------
    `image`: np.array
        The input image to the network

    `label`
        The label of the image

    `layers`: list
        A list of layers in the network, in the order they were applied

    Returns
    -------
    `output`: np.array
        The output of the network after forward propagation

    `loss`: float
        The loss (cross-entropy) of the network after forward propagation

    `accuracy`: int
        The accuracy of the network after forward propagation
    '''

    # Normalize the input image
    output = image/255.

    # Forward propagate through each layer
    for layer in layers:
        output = layer.forward_prop(output)

    # Compute loss (cross-entropy) and accuracy
    loss = -np.log(output[label])
    accuracy = 1 if np.argmax(output) == label else 0

    return output, loss, accuracy


def CNN_backprop(gradient, layers, alpha=0.05):
    """
    Backpropagates the given gradient through the layers of a convolutional neural network.

    Parameters
    ----------
    `gradient`: ndarray
        The gradient to backpropagate.

    `layers`: list
        A list of layers in the network, in the order they were applied.
    
    `alpha`: float, optional
        The learning rate for the backpropagation. Defaults to 0.05.

    Returns
    -------
    `ndarray`: The gradient after backpropagation through all layers.
    """

    # Initialize the gradient to be backpropagated
    grad_back = gradient

    # Backpropagate through each layer in reverse order
    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer, FullyConnectedLayer]:
            grad_back = layer.back_prop(grad_back, alpha)
        elif type(layer) == PoolingLayer:
            grad_back = layer.back_prop(grad_back)

    return grad_back


def CNN_training(image, label, layers, alpha=0.05):
    '''
    Train a convolutional neural network.

    Parameters
    ----------
    `image`: np.array
        The input image to the network

    `label`: int
        The true label of the image

    `layers`: list
        A list of layers in the network, in the order they were applied

    `alpha`: float
        The learning rate for the backpropagation. Defaults to 0.05.

    Returns
    -------
    `loss`: float
        The loss (cross-entropy) of the network after training

    `accuracy`: int
        The accuracy of the network after training
    '''

    # Forward step
    output, loss, accuracy = CNN_forward(image, label, layers)

    # Initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1/output[label]

    # Backprop step
    gradient_back = CNN_backprop(gradient, layers, alpha)

    return loss, accuracy

# if __name__ == "__main__":

#     cl = ConvolutionLayer(5, 3)
#     filename = "image.jpg"
#     image = Image.open(filename).convert("L")
#     image.save(f"{filename}_greyscale.jpg")
#     image = np.asarray(image)
#     img = cl.forwardProp(image)

#     for i in range(len(img[0][0])):
#         plt.imshow(img[:,:,i], cmap='gray')
#         plt.savefig(f"{filename}_output_{i}.jpg", dpi=img.shape[0])
#         plt.show()
        
if __name__ == "__main__":
    # Define the layers of the CNN
    layers = [
        ConvolutionLayer(5, 3),
        PoolingLayer(2),
        FullyConnectedLayer(13*13*3, 100),
        FullyConnectedLayer(100, 20)
    ]

    # Load the image and label data

    filename = "image.jpg"
    image = Image.open(filename).convert("L")
    image.save(f"{filename}_greyscale.jpg")
    image = np.asarray(image)
    images, labels = load_data()

    # Train the CNN on the image data
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        loss, accuracy = CNN_training(image, label, layers)

        # Print the loss and accuracy for each image
        print(f"Image {i+1}: Loss = {loss:.4f}, Accuracy = {accuracy}")
