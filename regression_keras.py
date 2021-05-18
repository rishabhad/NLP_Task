#from keras.models import  Sequential
import pandas as pd
from keras.layers import Activation,Dense
from keras import optimizers
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing_data=pd.read_csv('Housing.csv')
#print(housing_data.isnull().sum())
# We do not have any null data

for column in ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']:
    housing_data[column] = housing_data[column].map(
        {'yes': 1, 'no': 0})

y=housing_data['price'].values
X=housing_data.drop(['price','furnishingstatus'],axis=1).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float32))
X_test = s_scaler.transform(X_test.astype(np.float32))


model = Sequential()
model.add(Dense(10, input_shape = (11,)))
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(1))

sgd = optimizers.SGD(lr = 0.01)    # stochastic gradient descent optimizer

model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])
model.fit(X_train, y_train, batch_size = 50, epochs = 50, verbose = 1)
model.summary()

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('mse: ', results[1])





