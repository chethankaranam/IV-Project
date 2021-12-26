from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Create your views here.
class LinearRegression:
    '''
    A class which implements linear regression model with gradient descent.
    '''
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None
        self.loss = []
        
    @staticmethod
    def _mean_squared_error(y, y_hat):
        '''
        Private method, used to evaluate loss at each iteration.
        
        :param: y - array, true values
        :param: y_hat - array, predicted values
        :return: float
        '''
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i]) ** 2
        return error / len(y)
    
    def fit(self, X, y):
        '''
        Used to calculate the coefficient of the linear regression model.
        
        :param X: array, features
        :param y: array, true values
        :return: None
        '''
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            # Line equation
            y_hat = np.dot(X, self.weights) + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)
            
            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))
            
            val = self.learning_rate*partial_w
            # Update the coefficients
            self.weights -= val
            self.bias -= self.learning_rate * partial_d
        
        
    def predict(self, X):
        '''
        Makes predictions using the line equation.
        
        :param X: array, features
        :return: array, predictions
        '''
        return np.dot(X, self.weights) + self.bias

def home(request):
    return render(request,'index_pro.html')

def predict(request):
    df = pd.read_csv(r"C:\Users\SAI\projects\djangoTuts\Project\points1_1.csv")
    x = df.iloc[::,:4].values
    y = df['target'].values.reshape(-1,1)
    s = StandardScaler()
    x = s.fit_transform(x)
    y = s.fit_transform(y).flatten()
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.9)
    model = LinearRegression()
    model.fit(x_train, y_train)

    input_arr = [float(request.POST['sem1']),float(request.POST['sem2']),float(request.POST['sem3']),float(request.POST['sem4'])]
    pred_sgpa = model.predict(input_arr)
    pred_cgpa = (sum(input_arr)+pred_sgpa)/5
    return render(request,'predict.html',{"sgpa":pred_sgpa.round(2),"cgpa":pred_cgpa.round(2)})
