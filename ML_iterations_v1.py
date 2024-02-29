import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_numeric.csv")

df = df.dropna()
filtered_data = df[df['current_size'] > 0.5].copy()
data_array = df.to_numpy()
filtered_data_array = filtered_data.to_numpy()
print(filtered_data_array.shape)

max_iteration = 1000
num_CV = 100

def regression(learning_rate, training_size, iterations):

    np.random.shuffle(data_array) 
    df_training = data_array[:training_size,:]
    df_cv = filtered_data_array[:num_CV,:]

    data_x = np.c_[np.ones(training_size, dtype=float), df_training[:, [1,2,3,4]]]
    data_y = df_training[:, 0]
    # for i in range(int(training_size)):
    #     plt.scatter(df_training[i, 3],data_y[i])
    # plt.show() 

    cv_x = np.c_[np.ones(num_CV, dtype=float), df_cv[:, [1,2,3,4]]]
    cv_y = df_cv[:, 0]

    theta = np.zeros([5, 1])
    thetas = np.zeros([iterations, 5])
    cost_functions = np.zeros([iterations, 2])

    for i in range(int(iterations)):
        prediction = data_x @ theta
        errors = prediction.T - data_y
        theta[0] -= learning_rate / training_size * np.sum(errors)
        for j in range(4):
            theta[j] -= learning_rate / training_size * np.sum(errors * data_x[:,j])
        thetas[i] = theta.T.copy()
        cost_functions[i,0] = (np.sum(np.square((cv_x @ theta).T - cv_y))) / 2 / num_CV
        cost_functions[i,1] = (np.sum(np.square((data_x @ theta).T - data_y))) / 2 / training_size
    print(theta)
    return cost_functions


def graph(x):  
    y = regression(0.1, 10000, max_iteration)
    y1 = y[:, 0]
    y2 = y[:, 1]

    plt.yscale('log')

    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value")
    plt.plot(x, y1, 'b')  
    plt.plot(x, y2, 'r')
    plt.show() 


domain = np.zeros(max_iteration)
for i in range(max_iteration):
    domain[i] = i+1
graph(domain)
