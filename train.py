from script import neuralNetwork
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

#create a dataset
X, Y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).T
X_test = scaler.transform(X_test).T
Y_train = Y_train.reshape(1, Y_train.shape[0])
Y_test = Y_test.reshape(1, Y_test.shape[0])

#train the model
nn = neuralNetwork(input_layer=2, output_layer=1, hidden_layer=4)
nn.train(X_train, Y_train, learning_rate=0.01, epochs=10000)

#predict on the test set
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == Y_test)
print(f"Total precision : {accuracy * 100}%")