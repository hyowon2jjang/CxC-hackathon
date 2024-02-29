import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\wild_fire_model_v4.csv")
theta = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\theta.csv")

columns_to_get = ['current_size','fire_spread_rate','temperature','relative_humidity','wind_speed','fire_fighting_start_size','duration_from_reported_to_dispatch', 'fire_spread_rate_squared',	'temperature_squared',	'relative_humidity_squared',	'wind_speed_squared',	'fire_fighting_start_size_squared']

filtered_data_A = df[df['current_size'] < 0.1].copy()
filtered_data_B = df[(df['current_size'] > 0.1) & (df['current_size'] < 4)].copy()
filtered_data_C = df[(df['current_size'] > 4) & (df['current_size'] < 40)].copy()
filtered_data_D = df[(df['current_size'] > 40) & (df['current_size'] < 200)].copy()
filtered_data_E = df[df['current_size'] > 200].copy()

print(filtered_data_A.shape)
print(filtered_data_B.shape)
print(filtered_data_C.shape)
print(filtered_data_D.shape)
print(filtered_data_E.shape)

# precision
data_array_A = filtered_data_A.to_numpy()
data_array_B = filtered_data_B.to_numpy()
data_array_C = filtered_data_C.to_numpy()
data_array_D = filtered_data_D.to_numpy()
data_array_E = filtered_data_E.to_numpy()

theta_array = theta.to_numpy()
flat_theta_array = theta_array.flatten()

prediction_A = np.exp(data_array_A[: , 2:13] @ flat_theta_array.astype(float))
right_prediction_A = prediction_A[prediction_A < 0.1]
prediction_B = np.exp(data_array_B[: , 2:13] @ flat_theta_array.astype(float))
right_prediction_B = prediction_B[(prediction_B > 0.1) & (prediction_B < 4)].copy()
prediction_C = np.exp(data_array_C[: , 2:13] @ flat_theta_array.astype(float))
right_prediction_C = prediction_C[(prediction_C > 4) & (prediction_C < 40)].copy()
prediction_D = np.exp(data_array_D[: , 2:13] @ flat_theta_array.astype(float))
right_prediction_D = prediction_D[(prediction_D > 40) & (prediction_D < 200)].copy()
prediction_E = np.exp(data_array_E[: , 2:13] @ flat_theta_array.astype(float))
right_prediction_E = prediction_E[prediction_E > 200]

# recall
df_array = df.to_numpy()
prediction_df = np.c_[np.exp(df_array[:, 2:13] @ theta_array.astype(float)), df_array[:, 13:]]

predicted_to_A = prediction_df[prediction_df[:,0] <= 0.1]
right_predicted_A = predicted_to_A[predicted_to_A[:,1] == 1]
predicted_to_B = prediction_df[(prediction_df[:,0] > 0.1) & (prediction_df[:,0] <= 4)]
right_predicted_B = predicted_to_B[predicted_to_B[:,2] == 1]
predicted_to_C = prediction_df[(prediction_df[:,0] > 4) & (prediction_df[:,0] <= 40)]
right_predicted_C = predicted_to_C[predicted_to_C[:,3] == 1]
predicted_to_D = prediction_df[(prediction_df[:,0] > 40) & (prediction_df[:,0] <= 200)]
right_predicted_D = predicted_to_D[predicted_to_D[:,4] == 1]
predicted_to_E = prediction_df[prediction_df[:,0] > 200]
right_predicted_E = predicted_to_E[predicted_to_E[:,5] == 1]

precision_A = right_prediction_A.shape[0] / filtered_data_A.shape[0] *100
precision_B = right_prediction_B.shape[0] / filtered_data_B.shape[0] *100
precision_C = right_prediction_C.shape[0] / filtered_data_C.shape[0] *100
precision_D = right_prediction_D.shape[0] / filtered_data_D.shape[0] *100
precision_E = right_prediction_E.shape[0] / filtered_data_E.shape[0] *100

recall_A = right_predicted_A.shape[0] / (predicted_to_A.shape[0] + 1) * 100
recall_B = right_predicted_B.shape[0] / (predicted_to_B.shape[0] + 1) * 100
recall_C = right_predicted_C.shape[0] / (predicted_to_C.shape[0] + 1) * 100
recall_D = right_predicted_D.shape[0] / (predicted_to_D.shape[0] + 1) * 100
recall_E = right_predicted_E.shape[0] / (predicted_to_E.shape[0] + 1) * 100


print("A precision:", precision_A, "%")
print("B precision:", precision_B, "%")
print("C precision:", precision_C, "%")
print("D precision:", precision_D, "%")
print("E precision:", precision_E, "%")

print(" ")

print("A recall:", recall_A, "%")
print("B recall:", recall_B, "%")
print("C recall:", recall_C, "%")
print("D recall:", recall_D, "%")
print("E recall:", recall_E, "%")

print(" ")

print("F1 score, class A:", (2 * precision_A * recall_A) / (precision_A + recall_A + 0.01))
print("F1 score, class B:", (2 * precision_B * recall_B) / (precision_B + recall_B + 0.01))
print("F1 score, class C:", (2 * precision_C * recall_C) / (precision_C + recall_C + 0.01))
print("F1 score, class D:", (2 * precision_D * recall_D) / (precision_D + recall_D + 0.01))
print("F1 score, class E:", (2 * precision_E * recall_E) / (precision_E + recall_E + 0.01))