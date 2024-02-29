import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\data.csv")

data_array = df.to_numpy()


def regression(learning_rate, training_size, iterations):

    df_training = data_array[:training_size,:]
    # df_cv = data_array[15000:,:]

    data_x = np.c_[np.ones(training_size, dtype=float), df_training[:, [5, 6]]]
    data_y = df_training[:, 3]

    theta = np.zeros([3, 1])
    prev_theta = theta.copy()

    for i in range(iterations):
        for j in range(3):
            theta[j-1] = theta[j-1] - learning_rate / training_size * (((data_x @ prev_theta).T - data_y) @ data_x[:, j])
        prev_theta = theta.copy()

    return (np.sum(np.square((data_x @ theta).T - data_y)))



def graph(x):  
    y = np.zeros(1000)
    for i in range(1000):
        y[i] = regression(x[i], 100, 1000)

    plt.yscale('log')

    plt.plot(x,y)  
    plt.show() 

domain = np.zeros(1000)
domain[0] = 0.0000000000001
for i in range(1, 1000):
    domain[i] = domain[i-1] * 1.01


graph(domain)