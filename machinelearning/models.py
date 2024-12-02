import torch
from torch import no_grad, stack
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Module



"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension '1 x dimensions'
        """
        super(PerceptronModel, self).__init__()

        self.w = Parameter(ones(1, dimensions))  # Initialize weights with ones of shape (1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return torch.tensordot(x, self.w, dims=([1],[1]))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)

        return 1 if score.item() >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        while True:
            error_count = 0
            for batch in dataloader:
                features, labels = batch['x'], batch['label']
                prediction = self.get_prediction(features)
                if prediction != labels.item():
                    self.w.data += (labels.item() * features).squeeze(0)
                    error_count += 1
            if error_count == 0:
                break




class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        super().__init__()
        # Initialize your model parameters here
        self.layer1 = nn.Linear(1, 250)  # Input layer to first hidden layer
        self.layer2 = nn.Linear(250, 250)  # First hidden layer to second hidden layer
        self.layer3 = nn.Linear(250, 1)  # Second hidden layer to output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        predictions = self.forward(x)
        totloss = nn.MSELoss()(predictions, y)
        return totloss

    def train(self, dataset, learning_rate=0.001, batch_size=16, epochs=1000):
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                features, labels = batch['x'], batch['label']
                optimizer.zero_grad()

                losses = self.get_loss(features, labels)
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()
            


            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
            
            if avg_loss < 0.02:
                break

                        



class DigitClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        hidden_size = 128  # You can adjust this size
        output_size = 10
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def run(self, x):
        # Forward pass
        x1 = self.fc1(x)
        x2 = self.relu(x1)

        x3 = self.fc2(x2)
        return x3

    def get_loss(self, x, y):
        # Compute the loss
        logits = self.run(x)
        loss_fun = nn.CrossEntropyLoss()

        totloss = loss_fun(logits, y)
        return totloss

    def train(self, dataset, epochs=10, learning_rate=0.001):
        # Set up the DataLoader
        data_loader = DataLoader(dataset, batch_size=60, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                x = batch['x']
                y = batch['label']
                
                # x tensor
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                if x.ndim != 2:
                    x = x.view(-1, 28*28)
                
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.long)
                if y.ndim != 1:
                    y = y.argmax(dim=1)


                # Zero the gradients
                optimizer.zero_grad()
                
                logits = self.run(x)
                
                # Compute the loss
                loss = self.get_loss(x, y)
                # Backward pass
                loss.backward()
        
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')



class LanguageIDModel(Module):
    def __init__(self, num_layers=2, hidden_size=646):
        super(LanguageIDModel, self).__init__()
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = len(self.languages)

        # Initialize the RNN layer and the output layer
        self.rnn = nn.RNN(self.num_chars, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Weight initialization
        nn.init.xavier_uniform_(self.output_layer.weight)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def run(self, xs):
        batch_size = xs[0].size(0)
        h = self.init_hidden(batch_size).to(xs[0].device)

        # Process each character in the sequence using the RNN
        for t in range(len(xs)):
            x_t = xs[t]
            _, h = self.rnn(x_t.unsqueeze(0), h)

        # Pass the final hidden state through the output layer
        logits = self.output_layer(h[-1])
        return logits

    def get_loss(self, xs, y):
        logits = self.run(xs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y.argmax(dim=1))
        return loss

    def train(self, dataset, epochs=75, learning_rate=0.00085, batch_size=48):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                x = batch['x']
                y = batch['label']
                
                # Ensure x is a tensor and move dimensions
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                x = torch.movedim(x, 1, 0)
                
                # Ensure y is a tensor
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.long)

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')
        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"

    
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """


    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
 