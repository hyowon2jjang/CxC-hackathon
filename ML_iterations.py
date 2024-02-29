import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\data.csv")

data_array = df.to_numpy()
max_iteration = 10000

def regression(learning_rate, training_size, iterations):

    df_training = data_array[:training_size,:]
    df_cv = data_array[20000:22000,:]

    data_x = np.c_[np.ones(training_size, dtype=float), df_training[:, [24, 43]]]
    data_y = df_training[:, 3]

    cv_x = np.c_[np.ones(2000, dtype=float), df_cv[:, [24, 43]]]
    cv_y = df_cv[:, 3]

    theta = np.zeros([3, 1])
    prev_theta = theta.copy()
    thetas = np.zeros([iterations, 3])
    cost_functions = np.zeros(iterations)

    for i in range(int(iterations)):
        for j in range(3):
            theta[j-1] = theta[j-1] - learning_rate / training_size * (((data_x @ prev_theta).T - data_y) @ data_x[:, j])
        prev_theta = theta.copy()
        thetas[i] = theta.T.copy()
        # cost_functions[i] = (np.sum(np.square((cv_x @ theta).T - cv_y))) / 2 / training_size
        cost_functions[i] = (np.sum(np.square((data_x @ theta).T - data_y))) / 2 / training_size


    return cost_functions



def graph(x):  
    y = regression(0.0000000001, 10000, max_iteration)

    plt.yscale('log')

    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value")
    plt.plot(x, y)  
    plt.show() 



domain = np.zeros(max_iteration)
for i in range(max_iteration):
    domain[i] = i+1

graph(domain)
