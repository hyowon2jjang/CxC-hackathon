import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_model_v4.csv")
theta = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\theta_v4.csv")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

columns_to_get = ['current_size','fire_spread_rate','temperature','relative_humidity','wind_speed','fire_fighting_start_size','duration_from_reported_to_dispatch', 'fire_spread_rate_squared',	'temperature_squared',	'relative_humidity_squared',	'wind_speed_squared',	'fire_fighting_start_size_squared']

filtered_data_A = df[df['size_class_A'] == 1].copy()
filtered_data_B = df[df['size_class_B'] == 1].copy()
filtered_data_C = df[df['size_class_C'] == 1].copy()
filtered_data_D = df[df['size_class_D'] == 1].copy()
filtered_data_E = df[df['size_class_E'] == 1].copy()

# data_array_A = filtered_data_A.to_numpy()
# data_array_A = np.c_[np.ones(filtered_data_A.shape[0], dtype=float), data_array_A[:, 1:12]]
data_array_B = filtered_data_B.to_numpy()
data_array_B = np.c_[np.ones(filtered_data_B.shape[0], dtype=float), data_array_B[:, 1:12]]
data_array_C = filtered_data_C.to_numpy()
data_array_C = np.c_[np.ones(filtered_data_C.shape[0], dtype=float), data_array_C[:, 1:12]]
data_array_D = filtered_data_D.to_numpy()
data_array_D = np.c_[np.ones(filtered_data_D.shape[0], dtype=float), data_array_D[:, 1:12]]
data_array_E = filtered_data_E.to_numpy()
data_array_E = np.c_[np.ones(filtered_data_E.shape[0], dtype=float), data_array_E[:, 1:12]]

print(theta)
theta_array = theta.to_numpy()

# prediction_A = sigmoid(data_array_A @ theta_array.astype(float))
# right_prediction_A = prediction_A[data_array_A[:,13] == np.max(prediction_A)]
prediction_B = sigmoid(data_array_B @ theta_array.astype(float))
print(prediction_B)
right_prediction_B = prediction_B[data_array_B[:,14] == np.max(prediction_B[:])]
prediction_C = sigmoid(data_array_C @ theta_array.astype(float))
right_prediction_C = prediction_C[data_array_C[:,15] == np.max(prediction_C[:])]
prediction_D = sigmoid(data_array_D @ theta_array.astype(float))
right_prediction_D = prediction_D[data_array_D[:,16] == np.max(prediction_D[:])]
prediction_E = sigmoid(data_array_E @ theta_array.astype(float))
right_prediction_E = prediction_E[data_array_E[:,17] == np.max(prediction_E[:])]

# recall
# df_array = df.to_numpy()
# prediction_df = np.c_[np.exp(df_array[:, 2:13] @ theta_array.astype(float)), df_array[:, 13:]]
# # predicted_to_A = prediction_A[prediction_A < 0.1]
# # right_predicted_A = predicted_to_A[predicted_to_A['size'] == 1]
# predicted_to_B = prediction_B[prediction_B < 0.1]
# right_predicted_B = predicted_to_B[predicted_to_B['size'] == 1]
# predicted_to_C = prediction_C[prediction_C < 0.1]
# right_predicted_C = predicted_to_C[predicted_to_C['size'] == 1]
# predicted_to_D = prediction_D[prediction_D < 0.1]
# right_predicted_D = predicted_to_D[predicted_to_D['size'] == 1]
# predicted_to_E = prediction_E[prediction_E < 0.1]
# right_predicted_E = predicted_to_E[predicted_to_E['size'] == 1]


# print("A precision:", right_prediction_A.shape[0] / filtered_data_A.shape[0] *100, "%")
print("B precision:", right_prediction_B.shape[0] / filtered_data_B.shape[0] *100, "%")
print("C precision:", right_prediction_C.shape[0] / filtered_data_C.shape[0] *100, "%")
print("D precision:", right_prediction_D.shape[0] / filtered_data_D.shape[0] *100, "%")
print("E precision:", right_prediction_E.shape[0] / filtered_data_E.shape[0] *100, "%")

