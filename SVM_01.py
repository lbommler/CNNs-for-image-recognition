# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import timeit
import pprint

# Importing the dataset
#path = "C:/Users/LukasPC/OneDrive/Wichtige Dokumente/Udemy/Machine Learning AZ/Data/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)"
#os.chdir(path)
#dataset = pd.read_csv('Social_Network_Ads.csv')
#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, 4].values

#testing SVM on small data
WDPath = "C:/Users/LukasPC/OneDrive/Studium/Master/5 - WS 18 19/Machine Learning Seminar/Data"
os.chdir(WDPath) 
#data = pd.read_pickle("Features/data_2_1100.pkl")
cm_total = []

C_2d_range = [1e4]
gamma_2d_range = [1e-4]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        for counter in range(28,29):
            start_1 = timeit.default_timer()
            X = np.array(data[:(counter*144)])
            y = np.array(target[:(counter*144)])
            
            # Splitting the dataset into the Training set and Test set
            from sklearn.cross_validation import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            #print(X_train)
            
            # Feature Scaling
            #from sklearn.preprocessing import StandardScaler
            #sc = StandardScaler()
            #X_train = sc.fit_transform(X_train)
            #X_test = sc.transform(X_test)
                        
            # Fitting SVM to the Training set
            from sklearn import svm, metrics # support vector classification
            ## Radiale Basis
            #classifier = svm.SVC(kernel = 'rbf', C = C, gamma = gamma) #random_state to give same results
            ## linear case with SVM
            #classifier = svm.SVC(kernel = 'linear', C = 10000) #random_state to give same results
            ## linear case with LinearSVC (way faster than LinearSVC)
            classifier = svm.LinearSVC()
            classifier.fit(X_train, y_train)
            start_2, stop_1 = timeit.default_timer(), timeit.default_timer()
            
            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
            stop_2 = timeit.default_timer()
            
            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            print("for "+str(counter)+" images: "+str(round(stop_1 - start_1,1))+" modelling and "+str(round(stop_2 - start_2,1))+" predicting")
            print("Precision: "+str(round(cm[1][1]/(cm[1][1]+cm[0][1]),2))+" - Recall: "+str(round(cm[1][1]/(cm[1][1]+cm[1][0]),2)))
            print(cm)
            cm_total.append(counter)
            cm_total.append([C, gamma])
            cm_total.append(cm)
            
            print("Classification report for classifier %s:\n%s\n" 
                  % (classifier, metrics.classification_report(y_test, y_pred)))
# save confusion matrix in txt file
with open('your_file.txt', 'w') as f:
    for item in cm_total:
        f.write("%s\n" % item)

# Predict EvaluationImages
X_eval = sc.transform(data_test)
test_pred = classifier.predict(X_eval)

# Create evaluation_bitmaps.csv
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

eval_df = pd.DataFrame([concatenate_list_data(test_pred[i:i + 144]) for i in xrange(0, len(test_pred), 144)],
                        index = [picNr for picNr in range(5000, 5880)])
eval_df.to_csv("evaluation_bitmaps.csv", sep = ",", header = False)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()