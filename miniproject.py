import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
print(dataset_2.to_string()) #This line will print out your data

#Question 1
# Defining the variables
y = dataset_2.sum(axis=1)
X = dataset_2.drop(columns=['Date'])
# different combinations of bridges
bridge_Combinations = [
    ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'],
    ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge'],
    ['Brooklyn Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'],
    ['Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'],
]


top_score = 0
best_combo = None
#finds the r^2 for each combination and determines the best one
i = 0
while i < len(bridge_Combinations):
    combo = bridge_Combinations[i]
    combo_X = X[combo]
    train_comboX, test_comboX, train_comboY, test_comboY = train_test_split(combo_X, y, test_size=0.2, random_state=42)
    modelCombo = LinearRegression()
    modelCombo.fit(train_comboX, train_comboY)
    y_comboPred = modelCombo.predict(test_comboX)
    score = r2_score(test_comboY, y_comboPred)
    print(f'{combo}: R^2 score = {score: .4f}')
    if score > top_score:
        top_score = score
        best_combo = combo
    i += 1
print(f'Best combination: {best_combo}')

#Question 2 (Using a linear regression model)
weather = dataset_2[['High Temp', 'Low Temp', 'Precipitation']]
train_weatherX, test_weatherX, train_weatherY, weather_testY = train_test_split(weather, y, test_size=0.2, random_state=42)
modelWeather = LinearRegression()
modelWeather.fit(train_weatherX, train_weatherY)
weather_pred = modelWeather.predict(test_weatherX)
score = r2_score(weather_testY, weather_pred)
print(f'The R^2 score for this model is {score: .4f}')

#Question 3 (Using K-means clustering)
brooklyn = dataset_2['Brooklyn Bridge']
manhattan = dataset_2['Manhattan Bridge']
queensboro = dataset_2['Queensboro Bridge']
williamsburg = dataset_2['Williamsburg Bridge']
day_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
bike_counts = pandas.concat([brooklyn, manhattan, queensboro, williamsburg], axis=1)

# Scale the data (to improve accuracy of the k means cluster)
scaler = StandardScaler()
scaled_count = scaler.fit_transform(bike_counts)
# Perform the k means cluster (one cluster for each day of the week)
kmeans = KMeans(n_clusters=7, random_state=0).fit(scaled_count)
for i in range(7):
    centroid = scaler.inverse_transform(kmeans.cluster_centers_[i].reshape(1, -1))
    print("Centroid for ", day_of_week[i],":", centroid)
# Predict the day of the week
recent_Data = bike_counts.iloc[-1].values.reshape(1,-1)
scaled_Data = scaler.transform(recent_Data)
predicted_Cluster = kmeans.predict(scaled_Data)
predicted_Day = day_of_week[predicted_Cluster[0]]
print("The predicted day is:", predicted_Day)
