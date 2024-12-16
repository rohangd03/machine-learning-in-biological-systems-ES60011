import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, lr, epochs):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.lr = lr
        self.epochs = epochs

        # INITIALIZE THE WEIGHTS
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

        # BIAS
        self.b1 = np.zeros((1, self.hidden_layer_size))
        self.b2 = np.zeros((1, self.output_layer_size))

        
    # SIGMOID ACTIVATION FUNCTION
    def activation(self, X):
        z = 1/(1 + np.exp(-X))
        return z

    # FORWARD PASS
    def forward(self, X):
        self.hidden = self.activation(np.dot(X, self.W1) + self.b1)  
        # Hidden_layer_node = activation(X * W1 + b1)
        self.output = self.activation(np.dot(self.hidden, self.W2) + self.b2)  
        # Output_layer_node = activation(Z * W2 + b2)
        return self.output

    # BACKWARD PASS
    def back_prop(self, X, y):
        num = X.shape[0] # No. of ROWS in the DATASET 
        output = self.forward(X)            # (490, 4)
        error_output_layer = y - output     # (490, 4)
        
        delta_output_layer = output * (np.array(1) - output) * (y - output)
        self.W2 = self.W2 + self.lr * np.dot(self.hidden.T, delta_output_layer)  # (4, 4)
        
        delta_hidden_layer = self.hidden * (np.array(1) - self.hidden) * np.dot(delta_output_layer, self.W2.T)  # (490, 4)
        self.W1 = self.W1 + self.lr * np.dot(X.T, delta_hidden_layer)

        self.b2 = self.b2 + self.lr * np.sum(delta_output_layer, axis=0, keepdims=True)
        self.b1 = self.b1 + self.lr * np.sum(delta_hidden_layer, axis=0, keepdims=True)


    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.back_prop(X, y)

    def predict(self, X):
        return self.forward(X)


    # 5 FOLD CROSS VALIDATION
    def cross_validation_5(self, X, y):
        kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
        loss_history = np.zeros((5, self.epochs // 50))  
        # To store loss after every 50 epochs for each fold

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            # kfold.split(X) splits the X dataset into 5 folds
            # train_idx: Indices for the training set.
            # val_idx: Indices for the testing set.

            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]

            fold_loss = []
            for epoch in range(self.epochs):
                self.back_prop(X_train, y_train)
                
                if (epoch + 1) % 50 == 0:
                    loss = np.mean(np.square((y_test - self.forward(X_test))))
                    loss = loss.item()
                    fold_loss.append(loss)
                    print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {loss}")
            
            loss_history[fold] = fold_loss

        avg_loss = np.mean(loss_history, axis = 0)

        # Plot the loss vs epochs
        plt.plot(range(50, self.epochs + 1, 50), avg_loss, marker = 'o')
        plt.title('Loss vs Epochs (5-Fold Cross Validation)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    # 10 FOLD CROSS VALIDATION
    def cross_validation_10(self, X, y):
        kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
        loss_history = np.zeros((10, self.epochs // 50))  
        # To store loss after every 50 epochs for each fold

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]

            fold_loss = []
            for epoch in range(self.epochs):
                self.back_prop(X_train, y_train)
                
                if (epoch + 1) % 50 == 0:
                    loss = np.mean(np.square((y_test - self.forward(X_test))))
                    loss = loss.item()
                    fold_loss.append(loss)
                    print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {loss}")
            
            loss_history[fold] = fold_loss

        avg_loss = np.mean(loss_history, axis = 0)

        # Plot the loss vs epochs
        plt.plot(range(50, self.epochs + 1, 50), avg_loss, marker = 'o')
        plt.title('Loss vs Epochs (5-Fold Cross Validation)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()