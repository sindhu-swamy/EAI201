import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Prepare the house price data
df = pd.DataFrame({
    'Area': [1200, 1400, 1600, 1700, 1850],
    'Rooms': [3, 4, 3, 5, 4],
    'Distance': [5, 3, 8, 2, 4],
    'Age': [10, 3, 20, 15, 7],
    'Price': [120, 150, 130, 180, 170]
})

# Step 2: Fit regressions for each feature alone
X_area = df[['Area']]
X_rooms = df[['Rooms']]
X_distance = df[['Distance']]
X_age = df[['Age']]
y = df['Price']

model_area = LinearRegression().fit(X_area, y)
model_rooms = LinearRegression().fit(X_rooms, y)
model_distance = LinearRegression().fit(X_distance, y)
model_age = LinearRegression().fit(X_age, y)

print("R^2 with Area:", model_area.score(X_area, y))
print("R^2 with Rooms:", model_rooms.score(X_rooms, y))
print("R^2 with Distance:", model_distance.score(X_distance, y))
print("R^2 with Age:", model_age.score(X_age, y))

# Step 3: Try combinations with the best single feature (Rooms)
X_rooms_area = df[['Rooms', 'Area']]
X_rooms_distance = df[['Rooms', 'Distance']]
X_rooms_age = df[['Rooms', 'Age']]

model_rooms_area = LinearRegression().fit(X_rooms_area, y)
model_rooms_distance = LinearRegression().fit(X_rooms_distance, y)
model_rooms_age = LinearRegression().fit(X_rooms_age, y)

print("R^2 with Rooms + Area:", model_rooms_area.score(X_rooms_area, y))
print("R^2 with Rooms + Distance:", model_rooms_distance.score(X_rooms_distance, y))
print("R^2 with Rooms + Age:", model_rooms_age.score(X_rooms_age, y))
