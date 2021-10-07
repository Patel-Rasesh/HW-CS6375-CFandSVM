from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import timeit

# Following are the parameters which can be changed
# class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100), activation='relu', *, solver='adam', 
# alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, 
# shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
# nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

start = timeit.default_timer()

X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)

X = X/255
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

#model = MLPClassifier(max_iter=1, warm_start=True)
#model = MLPClassifier(max_iter=200, activation="relu", alpha=0.01, hidden_layer_sizes=(40,), batch_size=200, learning_rate_init=0.01, random_state=None, 
 #                       tol=1e-4, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, learning_rate="adaptive", solver="adam")
model = MLPClassifier(max_iter=15000, activation="logistic", alpha=0.01, hidden_layer_sizes=(100,),random_state=None, tol=1e-4, solver="lbfgs", max_fun=15000, warm_start=False)

parameters={
    'hidden_layer_sizes' : [(30,), (35), (40,)],
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'alpha': [0.0001, 0.01, 0.001],
    'solver' : ['sgd', 'adam'],
    'activation': ["logistic", "relu", "tanh", "identity"]
}

#To find the best parameters
#classifier = GridSearchCV(model, param_grid=parameters, n_jobs=-1, cv=5)
#classifier.fit(X_train, y_train)
#print(classifier.best_params_)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Error rate ", (1-metrics.accuracy_score(y_test, y_pred))*100)

stop = timeit.default_timer()
print("Runtime is ",stop-start)
