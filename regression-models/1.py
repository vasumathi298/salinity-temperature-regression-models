import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


balance_data = pd.read_csv('bottle.csv')
balance_data.fillna(0, inplace=True)
X = balance_data.values[:, 0:3]
Y = balance_data.values[:, 2]
print(len(X),len(Y))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.003, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test,y_test)
print(accuracy*100,'%')

y_pred = regressor.predict(X_test)
plt.plot(y_test,label='actual',ls=('dashed'),color='black')
plt.plot(y_pred,label='predicted',color='black')
plt.legend()
plt.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Variance score:',metrics.explained_variance_score(y_test,y_pred))
print('Maximum residual  error:',metrics.max_error(y_test,y_pred))
print('Median absolute error regression loss:',metrics.median_absolute_error(y_test,y_pred))
print('Regression score function:',metrics.r2_score(y_test,y_pred))
