import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\data.csv")

data_array = df.to_numpy()
max_iteration = 100

def regression(training_size):

    np.random.shuffle(data_array) 
    df_training = data_array[:training_size,:]
    df_cv = data_array[20000:22000,:]

    for i in range(int(1000)):
        plt.scatter(df_training[i, 3],data_y[i])

    data_x = np.c_[np.ones(training_size, dtype=float), df_training[:, 3]]
    data_y = df_training[:, 3]
    # for i in range(int(training_size)):
    #     plt.scatter(df_training[i, 3],data_y[i])
    # plt.show() 

    cv_x = np.c_[np.ones(2000, dtype=float), df_cv[:, 24]]
    cv_y = df_cv[:, 3]

    theta = np.linalg.pinv(data_x.T @ data_x)  @ data_x.T @ data_y

    return (data_x @ theta) - data_y



def graph(x):  
    y = regression(1000)
    plt.yscale('log')

    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value")
    plt.plot(x, y)  

    xpoints = np.array([0, 8])
    ypoints = np.array([3, 10])

    plt.plot(xpoints, ypoints)
    plt.show() 

graph()
