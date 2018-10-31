import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dataset_filepath = r'C:\Users\walkerl\Documents\ML_tutorial\iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(dataset_filepath, names=names)

# print(dataset.groupby('class').size())
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=True, sharey=True)
# dataset.hist(sharex=True, sharey=True)
# scatter_matrix(dataset)
# plt.show()

# Split the dataset in 20/80
X = dataset.values[:,0:4]
Y = dataset.values[:,4]
validation_size = 0.20
seed = 7  # Braucht es den seed=?

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('LogReg', LogisticRegression()))
models.append(('LinDiscAn', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('ClassARegTrees', DecisionTreeClassifier()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('SupVecMa', SVC()))


# Model evaluation:
results = []
model_names = []

for model_name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    results.append(cv_results)
    names.append(model_name)
    msg = "%s: %f (%f)" % (model_name, cv_results.mean(), cv_results.std())

# Make predictions on validatoin dataset

knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


