from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd
from src.logger.auto_logger import autolog


class ModelFinder:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test= X_test
        self.y_test = y_test
        self.dtc = DecisionTreeClassifier()
        self.rfc = RandomForestClassifier()
        self.knn = KNeighborsClassifier()


    def getBestparamsDecisionTree(self):
        try:
            autolog("Initializing with different combination of parameters")
            self.params_dtc = {'criterion'    : ['gini', 'entropy'],
                                'splitter'    : ["best", "random"], 
                                'max_depth'   : range(2, 6, 1), 
                                'max_features': ['auto' , 'log2']   
                                }

            autolog("Initializing GridSearchCV")
            self.grid = GridSearchCV(self.dtc, self.params_dtc, cv=5, verbose = True)
            
            autolog("Finding best parameters for Decision Tree Classifier")
            self.grid.fit(self.X_train, self.y_train)

            autolog(message="Best parameters for Decision Tree Classifier: {}".format(self.grid.best_params_))
            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']

            autolog("Creating an object for Decision Tree with best parameters")
            self.dtc = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, 
                                                max_depth=self.max_depth, max_features=self.max_features)
            
            autolog("Fitting the model with best parameters")
            self.dtc.fit(self.X_train, self.y_train)
            autolog("Model Fitted....")

        except Exception as e:
            autolog("Error in finding best parameters for Decision Tree Classifier: {}".format(e))


    def getBestparamsRandomForest(self):
        try:
            autolog("Initializing with different combination of parameters")
            self.params_rfc = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                'criterion'    : ['gini', 'entropy'],
                                'max_depth'   : range(2, 6, 1), 
                                'max_features': ['auto' , 'log2']   
                                }

            autolog("Initializing GridSearchCV")
            self.grid = GridSearchCV(self.rfc, self.params_rfc, cv=5, verbose = True)

            autolog("Finding best parameters for Random Forest Classifier")
            self.grid.fit(self.X_train, self.y_train)

            autolog(message="Best parameters for Random Forest Classifier: {}".format(self.grid.best_params_))
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']

            autolog("Creating an object for Random Forest with best parameters")
            self.rfc = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                                max_depth=self.max_depth, max_features=self.max_features)

            autolog("Fitting the model with best parameters")
            self.rfc.fit(self.X_train, self.y_train)
            autolog("Model Fitted....")

        except Exception as e:
            autolog("Error in finding best parameters for Random Forest Classifier: {}".format(e))



    def getBestparamsKNN(self):
        try:
            autolog("Initializing with different combination of parameters")
            self.params_knn = {'n_neighbors'      : range(1, 11, 1),
                                'leaf_size'       : list(range(10, 30, 2)),
                                'weights'         : ['uniform', 'distance'],
                                'algorithm'       : ['auto', 'ball_tree', 'kd_tree', 'brute']
                                }

            autolog("Initializing GridSearchCV")
            self.grid = GridSearchCV(self.knn, self.params_knn, cv=5, verbose = True)

            autolog("Finding best parameters for KNN Classifier")
            self.grid.fit(self.X_train, self.y_train)

            autolog(message="Best parameters for KNN Classifier: {}".format(self.grid.best_params_))
            self.n_neighbors = self.grid.best_params_['n_neighbors']
            self.leaf_size = self.grid.best_params_['leaf_size']
            self.weights = self.grid.best_params_['weights']
            self.algorithm = self.grid.best_params_['algorithm']

            autolog("Creating an object for KNN with best parameters")
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, leaf_size=self.leaf_size,
                                                weights=self.weights, algorithm=self.algorithm) 
            autolog("Fitting the model with best parameters")
            self.knn.fit(self.X_train, self.y_train)
            autolog("Model Fitted....")

        except Exception as e:
            autolog(message="Error in finding best parameters for KNN Classifier: {}".format(e))

        
        def getBestModel(self):
            autolog("Finding best model")

            try:
                self.getBestparamsKNN(self.X_train,self.y_train,self.X_test,self.y_test)
                self.prediction_knn = self.knn.predict_proba(self.X_test)

                if len(self.y_test) == 1:
                    self.knn_score = accuracy_score(self.y_test, self.knn.predict(self.X_test))
                    autolog("Accuracy Score for KNN Classifier: {}".format(self.knn_score))
                else:
                    self.knn_score = roc_auc_score(self.y_test, self.prediction_knn)
                    autolog("AUC Score for KNN Classifier: {}".format(self.knn_score))

                self.getBestparamsDecisionTree(self.X_train,self.y_train,self.X_test,self.y_test)
                self.prediction_dtc = self.dtc.predict_proba(self.X_test)

                if len(self.y_test) == 1:
                    self.dtc_score = accuracy_score(self.y_test, self.dtc.predict(self.X_test))
                    autolog("Accuracy Score for Decision Tree Classifier: {}".format(self.dtc_score))
                else:
                    self.dtc_score = roc_auc_score(self.y_test, self.prediction_dtc)
                    autolog("AUC Score for Decision Tree Classifier: {}".format(self.dtc_score))

                self.getBestparamsRandomForest(self.X_train,self.y_train,self.X_test,self.y_test)
                self.prediction_rfc = self.rfc.predict_proba(self.X_test)

                if len(self.y_test) == 1:
                    self.rfc_score = accuracy_score(self.y_test, self.rfc.predict(self.X_test))
                    autolog("Accuracy Score for Random Forest Classifier: {}".format(self.rfc_score))
                else:
                    self.rfc_score = roc_auc_score(self.y_test, self.prediction_rfc)
                    autolog("AUC Score for Random Forest Classifier: {}".format(self.rfc_score))

                autolog("Comparing the models")
                if self.knn_score > self.dtc_score and self.knn_score > self.rfc_score:
                    autolog("KNN Classifier is the best model")
                    return "KNN",self.knn
                elif self.dtc_score > self.knn_score and self.dtc_score > self.rfc_score:
                    autolog("Decision Tree Classifier is the best model")
                    return "DecisionTreeClassifier",self.dtc
                elif self.rfc_score > self.knn_score and self.rfc_score > self.dtc_score:
                    autolog("Random Forest Classifier is the best model")
                    return "RandomForestClassifier",self.rfc
            except Exception as e:
                autolog("Error in finding best model: {}".format(e))


            