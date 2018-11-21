
"""
    A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from matplotlib import pyplot
# iris_data = load_iris() # load the iris dataset

flag_data = []
y = []
with open("data.data") as f:
    i = 0
    for line in f:
        a = line.split(',')
        flag_data.append(a[1:9] + a[18:28])
        y.append(a[8])


flag_data = np.array(flag_data)
y = np.array(y)

print('flag_data[0]', flag_data[0])
print('y[0]', y)

x = flag_data
y_ = y.reshape(-1, 1) # Convert data to a single column

# print("x, y", x, y_)

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
print("encoder")
print(y)
# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model

model = Sequential()

model.add(Dense(10, input_shape=(18,), activation='relu', name='fc1'))
model.add(Dense(100, activation='tanh', name='fc2'))
model.add(Dense(12, activation='sigmoid', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
hist = model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

plt.figure(figsize=(10,8))
plt.plot(hist.history['acc'], label='Accuracy')
plt.plot(hist.history['loss'], label='Loss')
plt.legend(loc='best')
plt.title('Training Accuracy and Loss')
plt.show()

# Test on unseen data

results = model.predict(test_x)
print(results)
