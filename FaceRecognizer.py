from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selecton import leaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti_data=fetch_olivetti_faces()

# there are 400 images- 10x40 (40 people -1 person has 10 images) -1 image = 64x64 pixels
features = olivetti_data.data
# we represent target variables (people) with integers (face ids)
targets = olivetti_data.target

print(targets)
print(targets.shape)
print(features.shape)
print(features)  # storing pixel intensities in numerical one D array
# every row in array is normalized since in between 1 and 0

fig, sub_plots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
print(sub_plots)
sub_plots = sub_plots.flatten()
print(sub_plots)

for unique_user_id in np.unique(targets):
    image_index = unique_user_id * 8
    sub_plots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap='gray')
    sub_plots[unique_user_id].set_xticks([])
    sub_plots[unique_user_id].set_yticks([])
    sub_plots[unique_user_id].set_title("Face id: %s" % unique_user_id)

plt.subtitle("The dataset (40 people")
plt.show()

# lets plot the 10 images for the first person (face id=0)

fig, sub_plots = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))

for j in range(10):
    sub_plots[j].imshow(features[j].reshape(64, 64), cmap="gray")
    sub_plots[j].set_xticks([])
    sub_plots[j].set_yticks([])
    sub_plots[j].set_title("Face id=0")

plt.show()


# split the original data-set (training and test set)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)

# lets try to find the optimal number of eigenvectors (principle components)

pca = PCA(n_components=100, whiten=True)
pca.fit(features)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# after we find the optimal 100 PCA numbers we can check the "eigenfaces"
# 1 principle component (eigenvector) has 4096 features
number_of_eigenfaces= len(pca.components_)
eigen_faces = pca.components_.reshape((number_of_eigenfaces, 64, 64))

fig, sub_plots = plt.subplots(nrows=18, ncols=10, figsize=(15, 15))

sub_plots = sub_plots.flatten()
print(sub_plots)

for i in range(number_of_eigenfaces):
    sub_plots[i].imshow(eigen_faces[1], cmap="gray")
    sub_plots[i].set_xticks([])
    sub_plots[i].set_yticks([])

plt.showplt


plt.figure(1, figsize=(12, 8))
plt.plot(pca.explained_variance_, linewidth=2)
plt.xlabel('Components')
plt.ylabel('Explained Variances')
plt.show()

# lets use the machine learning models

print(features.shape)
print(X_train_pca.shape)
print(X_test_pca.shape)

models = [("Logistic Regression", LogisticRegression()), ("Support Vector Machine", SVC()), ("Naive Bayes Classifier", GaussianNaiveBayes())]

for name, model in models:

    classifier_model = model
    classifier_model.fit(X_train_pca, y_train)

    y_predicted = classifier_model.predict(X_test_pca)
    print("Results with %s" % name)
    print("Accuracy score: %s" % (metrics.accuracy_score(y_test, y_predicted)))


