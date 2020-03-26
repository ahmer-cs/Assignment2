import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('dataset.csv')
print(data.shape)
data.head()

# Collecting X and Y
X = data.iloc[:,2:3].values
y = data.iloc[:,3:4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Head size vs Brain weight (Training set)')
plt.xlabel('Head size in cm3')
plt.ylabel('Brain weight in grams')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Head size vs Brain weight (Test sets)')
plt.xlabel('Head size in cm3')
plt.ylabel('Brain weight in grams')
plt.show()
