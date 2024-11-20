#Very Simple Linear ML from Scratch

import pandas as pd
import matplotlib.pyplot as plt

#Mean Squeared Error Function
def error_function(slope, bias, data) -> float:
    n = len(data)
    total_error = 0 #A Sum holder for the total error

    
    for i in range(n):
        x = data.iloc[i].YearsExperience
        y = data.iloc[i].Salary
        
        total_error += pow((y - ((slope * x) + bias)), 2) #Summation of (Labels - Predictions)^2
    
    return (total_error / n) #MSE


#Gradient Descent to minimize MSE and find the optimal parameter
def gradient_descent(current_slope, current_bias, data, learning_rate) -> tuple:
    n = len(data)
    slope_gradient = 0
    bias_gradient = 0
    
    for i in range(n):
        x = data.iloc[i].YearsExperience
        y = data.iloc[i].Salary
        
        slope_gradient += -(2/n) * x * (y - ((current_slope * x) + current_bias)) #Partial Derivative of the MSE w.r.t the slope
        bias_gradient += -(2/n) * (y - ((current_slope * x) + current_bias)) #Partial Derivative of the MSE w.r.t the bias
    
    slope = current_slope - (learning_rate * slope_gradient) #New Slope parameter after gradient descent
    bias = current_bias - (learning_rate * bias_gradient) #New Bias parameter after gradient descent
    
    return slope, bias


#Set the initial parameters and hyperparameters
slope = 0
bias = 0
learning_rate = 0.001
epochs = 1000

     
#Load the dataset using pandas    
data = pd.read_csv("Employment.csv")

for i in range(epochs):
    slope, bias = gradient_descent(slope, bias, data, learning_rate)

mean_squared_loss = error_function(slope, bias, data)

x = data['YearsExperience']
y = data['Salary']

print(f"The new slope = {slope}, the new Bias = {bias}")
print(f"Mean Squared Loss = {mean_squared_loss}")

prediction = (slope * x) + bias   #y = mx + b

#Plot the regression
plt.scatter(x, y, color = "blue", label = "Labels/Data")  #Actual Data on a scatterplot
plt.plot(x, prediction, color = "red", label = "Model Prediction") #Linear Regression
plt.xlabel("Experience (Years)") #x-axis
plt.ylabel("Salary in USD") #y-axis
plt.title("Employee Salary Prediction Model")
plt.legend()
plt.show()

