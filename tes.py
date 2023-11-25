class Layer:
    #init
    def __init__(self,n_inputs,n_outputs, activation, weight_init):
        self.n_inputs = n_inputs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.weight_init = weight_init
         # Initialize the weights and biases according to the activation function and weight initialization method
        if activation == "sigmoid" or activation == "softmax" or activation == "tanh":
            # Use Xavier initialization
            self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(1 / n_inputs)
            self.biases = np.zeros((1, n_outputs))
        elif activation == "relu":
            # Use He initialization
            self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
            self.biases = np.zeros((1, n_outputs))
        else:
            # Use random initialization
            self.weights = np.random.randn(n_inputs, n_outputs)
            self.biases = np.zeros((1, n_outputs))
        # Initialize the gradients and the cache for the forward and backward pass
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.z = None
        self.a = None
        # Forward pass the inputs through the layer and apply the activation function
    def forward(self, inputs):
        # Compute the linear combination of inputs, weights, and biases
        self.z = np.dot(inputs, self.weights) + self.biases
        print('forward inputs',np.shape(inputs))
        print('forward weight',np.shape(self.weights))
        # Apply the activation function
        if self.activation == "sigmoid":
            self.a = sigmoid(self.z)
        elif self.activation == "softmax":
            self.a = softmax(self.z)
        elif self.activation == "relu":
            self.a = relu(self.z)
        else:
            self.a = self.z
        # Return the outputs
        return self.a
    
    # Backward pass the error through the layer and compute the gradients
    def backward(self, inputs, error):
        # Apply the derivative of the activation function to the error
        if self.activation == "sigmoid":
            error = error * sigmoid_derivative(self.z)
        elif self.activation == "softmax":
            error = error * softmax_derivative(self.z)
        elif self.activation == "relu":
            error = error * relu_derivative(self.z)
        # Compute the gradients of weights and biases
        self.dweights = np.dot(inputs.T, error)
        print('backward dweights',np.shape(self.dweights))
        print('backward inputs', np.shape(inputs))
        print('backward error', np.shape(error))
        self.dbiases = np.sum(error, axis=0, keepdims=True)
        # Compute the error for the previous layer
        print('calc error...')
        error = np.dot(error, self.weights.T)
        print('sucess')
        # Return the error
        return error

    # Update the weights and biases with the gradients and the learning rate
    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dweights
        self.biases = self.biases - learning_rate * self.dbiases
