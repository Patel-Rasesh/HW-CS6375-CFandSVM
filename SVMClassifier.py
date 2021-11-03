from sklearn.datasets import fetch_openml
from sklearn import preprocessing, svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import timeit

from sklearn.utils.extmath import weighted_mode

start = timeit.default_timer()

X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)

X = X/255
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train_Scaled = preprocessing.scale(X_train)
y_train_Scaled = preprocessing.scale(y_train)

lab_enc = preprocessing.LabelEncoder()
#X_train_encoded = lab_enc.fit_transform(X_train)
#y_train_encoded = lab_enc.fit_transform(y_train)

parameters = {
    "C" : [10],
    "kernel" : ['rbf', 'linear'],
    #"kernel" : [, 'poly'],
    #'gamma' : [1e-3, 1e-4, 1e-5],
    #'coef0' : [0.0, 0.1, 0.5],
    #'tol' : [1e-4, 1e-3, 1e-5],
    #"class_weight" : [None, 'balanced'],
    "max_iter": [100, 1000]
    #'decision_function_shape' : ['ovo', 'ovr']
}

#Uncomment next four lines to run GridSearchCV
# model = svm.SVC()
# classifier = GridSearchCV(model, param_grid=parameters)
# classifier.fit(X_train_Scaled, y_train_Scaled)
# print(classifier.best_params_)

#Uncomment next four lines to run your model on test data
classifier = svm.SVC(C = 1, kernel = 'rbf', max_iter=1000, class_weight='balanced')
#classifier = svm.SVC(C = 12, kernel = 'sigmoid', max_iter=100, decision_function_shape='ovo', coef0=10)

#classifier.fit(X_train, y_train)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

print("Error rate = ", (1-metrics.accuracy_score(y_test, y_pred))*100)

stop = timeit.default_timer()
print("Runtime is ",stop-start)