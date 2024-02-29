import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\combined_wild_fire.csv")
test = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_mean.csv")

columns_to_get = ['current_size','fire_spread_rate','temperature','relative_humidity','wind_speed','fire_fighting_start_size','duration_from_reported_to_dispatch', 'fire_spread_rate_squared',	'temperature_squared',	'relative_humidity_squared',	'wind_speed_squared',	'fire_fighting_start_size_squared']

# new_df = pd.DataFrame()
# new_df[columns_to_get] = test[columns_to_get].copy()
# new_df['duration_from_reported_to_dispatch'] =pd.to_timedelta(new_df['duration_from_reported_to_dispatch'])
# new_df['duration_from_reported_to_dispatch'] = new_df['duration_from_reported_to_dispatch'].dt.total_seconds() / 60


test_array = df.to_numpy()

df['current_size'] = np.log(df['current_size'])
data_array = df.to_numpy()

print(data_array.shape)

max_iteration = 3000
num_CV = 3000

def regression(learning_rate, training_size, iterations):

    np.random.shuffle(data_array) 
    df_training = data_array[:training_size,:]
    df_cv = data_array[training_size:training_size+num_CV,:]

    data_x = np.c_[np.ones(training_size, dtype=float), df_training[:, 2:13]]
    data_y = df_training[:, 1]
    # for i in range(int(training_size)):
    #     plt.scatter(df_training[i, 3],data_y[i])
    # plt.show() 

    cv_x = np.c_[np.ones(num_CV, dtype=float), df_cv[:, 2:13]]
    cv_y = df_cv[:, 1]

    theta = np.zeros([12, 1])
    thetas = np.zeros([iterations, 12])
    cost_functions = np.zeros([iterations, 2])

    for i in range(int(iterations)):
        prediction = data_x @ theta
        errors = prediction.T - data_y
        theta[0] -= learning_rate / training_size * np.sum(errors)
        for j in range(12):
            theta[j-1] -= learning_rate / training_size * np.sum(errors * data_x[:,j])
        thetas[i] = theta.T.copy()
        cost_functions[i,0] = np.sum(np.square((cv_x @ theta).T - cv_y)) / 2 / num_CV
        cost_functions[i,1] = np.sum(np.square((data_x @ theta).T - data_y)) / 2 / training_size
    print(theta)
    np.savetxt("theta.csv", theta, delimiter=",")
    return cost_functions


def graph(x):  
    y = regression(0.000000000001, 18000, max_iteration)
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



