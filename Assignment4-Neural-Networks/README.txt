Name - ROHAN GHOSH DASTIDAR
Roll - 22CH30028
Email - rgdastidar2069@kgpian.iitkgp.ac.in, rgdastidar2069@gmail.com
-----------------------------------------------------------------------------------
FILES:
- 22CH30028__Neural__Networks.ipynb
- housing.csv
- NeuralNetwork.py
- README.txt

1. PROJECT TITLE
Implementation of a back propagation artificial neural network.

2. OVERVIEW
This project implements a backpropagation neural network with one input layer, one hidden layer, and one output layer using only the NumPy library. The model is trained on the Boston Housing dataset to predict housing prices.

3. LIBRARIES USED
numpy - For handling and manipulating dataset in order to implement the neural network algorithm on it
pandas - For loading and analyzing the Boston Housing dataset
matplotlib - For plotting the "Loss" vs "epochs" for each of the cross-validation algorithms
sklearn - For importing and applying K fold cross-validation

4. CONFIGURATION OF THE ARCHITECTURE
Input layer size = No. of features of the housing.csv dataset
Hidden layer size = user input
Output layer size = user input
Learning rate = user input
No. of epochs = 1000

5. HOW TO RUN THE CODE
- The "NeuralNetwork.py" file contains the custom neural network with the back propagation built using "Discriminative Linear Classifier"
- Execute the "22CH30028__Neural__Networks.ipynb" file
- Input the parameters for the neural network 
- The program outputs the loss values after every 50 epochs and the plot of Loss vs epochs for both 5 FOLD and 10 FOLD CROSS VALIDATION

6. RESULTS
- With the increase in epochs, the loss reduces and converges to a small value
- The 10 fold cross validation yields lowers losses compared to 5 fold cross validation
 
