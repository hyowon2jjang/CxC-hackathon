import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\data.csv")

color_map = {'A': 'lightyellow', 'B': 'yellow', 'C': 'orange', 'D': 'red', 'E': 'darkred'}

plt.figure(figsize=(8, 8)) 

background_img = plt.imread(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\FSA_map.png")
plt.imshow(background_img, extent=[-123, -99, 49, 60])

point_size = 5

for size, color in color_map.items():
    data = df[df['size_class'] == size]
    plt.scatter(data['fire_location_longitude'], data['fire_location_latitude'], color=color, s=point_size, label=size)

plt.xlabel('fire_location_latitude')
plt.ylabel('fire_location_longitude')
plt.title('Fire Locations Scatter Plot with Different Size Classes')
plt.legend(title='Size Class')


plt.grid(True)
plt.show()

