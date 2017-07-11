import pandas as pd
import numpy as np
from sklearn import neighbors, preprocessing, cross_validation

# read the csv file into our data variable
data = pd.read_csv('breast-cancer-wisconsin.data')

# delete the unwanted id column
data.drop(['id'], 1, inplace=True)

# make up for missing entries
data.replace('?', -9999, inplace=True)

# get our attributes and classes in place
X = np.array(data.drop(['class'], 1))
y = np.array(data['class'])

# split data into training and testing sections
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# initialize our classifier
knn = neighbors.KNeighborsClassifier()

# fit the classifier with the training data
knn.fit(X_train, y_train)

# calculating accuracy with test data
accuracy = knn.score(X_test, y_test)

# let's make a prediction
new_tests = np.array([[10, 10, 2, 3, 10, 2, 1, 8, 44], [10, 1, 12, 3, 1, 12, 1, 8, 12], [3, 1, 1, 3, 1, 12, 1, 2, 1]])
new_tests = new_tests.reshape(len(new_tests), -1)
prediction = knn.predict(new_tests)

# print out details
print "Accuracy: ", accuracy

print "Predictions:"
for pred in prediction:
	if pred == 2:
		print pred, "Benign"
	else: print pred, "Malignant"

