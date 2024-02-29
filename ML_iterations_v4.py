import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_model_v4.csv")
test = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_mean.csv")

columns_to_get = ['current_size','fire_spread_rate','temperature','relative_humidity','wind_speed','fire_fighting_start_size','duration_from_reported_to_dispatch']

new_df = pd.DataFrame()
new_df[columns_to_get] = test[columns_to_get].copy()
new_df['duration_from_reported_to_dispatch'] =pd.to_timedelta(new_df['duration_from_reported_to_dispatch'])
new_df['duration_from_reported_to_dispatch'] = new_df['duration_from_reported_to_dispatch'].dt.total_seconds() / 60


test_array = new_df.to_numpy()

df = df[df['current_size'] > 1]
df['current_size'] = np.log(df['current_size'])
data_array = df.to_numpy()

print(data_array.shape)

max_iteration = 10
num_CV = 300

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def regression(learning_rate, training_size, iterations):

    np.random.shuffle(data_array) 
    df_training = data_array[:training_size,:]
    df_cv = data_array[training_size:training_size+num_CV,:]

    data_x = np.c_[np.ones(training_size, dtype=float), df_training[:, 2:13]]
    data_y = df_training[:, 14:]

    cv_x = np.c_[np.ones(num_CV, dtype=float), df_cv[:, 2:13]]
    cv_y = df_cv[:, 14:]

    theta = np.random.rand(12, 4)
    cost_functions = np.zeros([iterations, 2])

    for i in range(int(iterations)):
        prediction = sigmoid(data_x @ theta)
        cv_prediction = sigmoid(cv_x @ theta)
        dcost = (data_x.T @ (prediction - data_y)) / training_size
        for k in range(4):
            for j in range(12):
                theta[j-1,k-1] -= learning_rate * dcost[j-1,k-1]
        cost_functions[i,0] = np.sum(-(cv_y * np.log(cv_prediction) + (1 - cv_y) * np.log(1.0000001 - cv_prediction)) / num_CV, axis=0)[0]
        cost_functions[i,1] = np.sum(-(data_y * np.log(prediction) + (1 - data_y) * np.log(1.0000001 - prediction)) / training_size, axis=0)[0]
        print(theta[1,1], cost_functions[i,1])

    np.savetxt("theta_v4.csv", theta, delimiter=",")
    return cost_functions

def graph(x):  
    y = regression(0.000001, 2000, max_iteration)
    y1 = y[:, 0]
    y2 = y[:, 1]

    plt.yscale('log')

    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value")
    plt.plot(x, y1, 'b', label='Cross Validation')
    plt.plot(x, y2, 'r', label='Training')
    plt.show()

domain = np.zeros(max_iteration)
for i in range(max_iteration):
    domain[i] = i+1
graph(domain)

