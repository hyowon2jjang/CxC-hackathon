import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox




df = pd.read_csv(r"C:\\Users\\hyowo\\OneDrive\\Documents\\programming\\2024 winter\\CxC hackathon\\data.csv")

df.scatter(x=df['current_size'], y=df['distance_from_water_source'], label='Data Points')

plt.xlabel('fire_location_latitude')
plt.ylabel('fire_location_longitude')
plt.title('Fire Locations Scatter Plot with Different Size Classes')
plt.legend(title='Size Class')

plt.grid(True)
plt.show()

