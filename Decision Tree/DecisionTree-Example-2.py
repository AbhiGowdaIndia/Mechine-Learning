from sklearn import tree

# You may hard code your data as given or to use a .csv file import csv then fetch your data from .csv file

# Assume we have two dimensional feature space with two classes we like distinguish
dataTable = [[2,9],[4,10],[5,7],[8,3],[9,1]]

dataLabels = ["Class A","Class A","Class B","Class B","Class B"]

# Declare our classifier
trained_classifier = tree.DecisionTreeClassifier()

# Train our classifier with data we have
trained_classifier = trained_classifier.fit(dataTable,dataLabels)

# We are done with training, so it is time to test it!
someDataOutOfTrainingSet = [[4,11]]
label = trained_classifier.predict(someDataOutOfTrainingSet)

# Show the prediction of trained classifier for data [11,2]
print(label[0])