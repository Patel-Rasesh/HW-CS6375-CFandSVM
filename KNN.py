from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import timeit

start = timeit.default_timer()

X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)

X = X/255
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

parameters = {
    "n_neighbors" : [10, 20],
    "weights" : ['uniform', 'distance'],
    "algorithm" : ['auto','kd_tree'],
    "leaf_size" : [20, 30],
    "p" : [1, 2],
    "metric" : ['euclidean', 'manhattan'],
    #"metric_params" : []
}
#Uncomment these lines for GridSearchCV
# classifier = GridSearchCV(model, param_grid=parameters, cv=3)
# classifier.fit(X_train, y_train)
# print(classifier.best_params_)

#Uncomment these lines to test the model
#model = KNeighborsClassifier(weights='distance', n_neighbors=20, n_jobs=-1, p=5)
#model = KNeighborsClassifier(weights='distance', algorithm='kd_tree', leaf_size = 50, n_neighbors=20, n_jobs=-1, p=10)
#model = KNeighborsClassifier(weights='distance', algorithm='kd_tree', leaf_size = 10, n_neighbors=20, n_jobs=-1, p=3)
#model = KNeighborsClassifier(weights='distance', algorithm='ball_tree', leaf_size = 50, n_neighbors=20, n_jobs=-1, p=3)
model = KNeighborsClassifier(weights='distance', algorithm='ball_tree', leaf_size = 10, n_neighbors=20, n_jobs=-1, p=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Error rate = ", (1-metrics.accuracy_score(y_test, y_pred))*100)

stop = timeit.default_timer()
print("Runtime is ",stop-start)
