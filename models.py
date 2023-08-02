from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import pickle

def log_model(df, test_perc, target_var, scale_lst, show_coef = False, apply_smote = False):
    '''
    fitting the logistic regression for given dataset
    input:
        df (pandas dataframe): the dataframe with features and outcome
        test_perc (float): percentage of test data
        target_var (str): name of target variable
        scale_lst (list): list of name of variables that will be scaled
        show_coef (boolean): whether to show the coefficient associated with each variable
    '''
    X = df.drop(columns = target_var)
    y = df.loc[:, target_var]
    
    #scaling continuous
    ct = ColumnTransformer([
        ('standarized', StandardScaler(),  scale_lst)
    ], remainder='passthrough')
    X_scaled = ct.fit_transform(X)
    
    if apply_smote:
        X_scaled, y = SMOTE().fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= test_perc, random_state=1)
    
    #fitting model
    log_reg = LogisticRegression(random_state=1, solver = 'liblinear').fit(X_train, y_train)
    y_predicted_training = log_reg.predict(X_train)
    y_predicted_test = log_reg.predict(X_test)
    
    #getting the accuracy
    training_score = log_reg.score(X_train, y_train)
    test_score = log_reg.score(X_test, y_test)
    scores = cross_val_score(log_reg, X_train, y_train, cv=5)
    print('accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    print('Cross-Validation Accuracy Scores', scores)
    print('Training Score:', training_score)
    print('Testing Score:', test_score)
    print('Classication Report:') 
    print(classification_report(y_predicted_test, y_test))
    
    if show_coef:
        col = ['_'.join(var.split('_')[2:]) for var in ct.get_feature_names_out()]
        lst = log_reg.coef_[0]
        print(log_reg.intercept_[0])
        coef_lst = []
        for var, coef in zip(col, lst):
            coef_lst.append((var, coef))
        
        coef_lst.sort(key = lambda x : x[1])
        for var, coef in coef_lst:
            print(var, ':', coef)
    
    RocCurveDisplay.from_estimator(log_reg, X_test, y_test)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], color="b", marker="x")
    plt.show()

def decision_tree_model(df, test_perc, target_var, m_depth, save_graph = False, apply_smote = False):
    '''
    Fitting the data into a decision tree model
    Input:
        df: pd dataframe
        test_perc: percentage of testing sample
        target_var: target label of decision tree
        m_depth: maximum depth of decision tree
        save_graph: whether to generate and save the decision tree graph
        apply_smote: whether to apply SMOTE
    '''
    X = df.drop(columns = [target_var])
    y = df.loc[:,target_var]
    if apply_smote:
        X, y = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_perc, random_state=1)
    
    #fitting the tree
    tree_pre = tree.DecisionTreeClassifier(max_depth = m_depth)
    tree_pre = tree_pre.fit(X_train, y_train)
    y_pred_train = tree_pre.predict(X_train)
    y_pred_test = tree_pre.predict(X_test)
    
    #printing result
    print('train accuracy', accuracy_score(y_train, y_pred_train),
      'test accuracy',  accuracy_score(y_test, y_pred_test))
    accuracy = accuracy_score(y_test, y_pred_test)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    if save_graph:
        plt.figure()
        tree.plot_tree(tree_pre, feature_names= X.columns, filled = True)
        plt.savefig('LBW.pdf', format = 'pdf',bbox_inches = "tight")
    
    RocCurveDisplay.from_estimator(tree_pre, X_test, y_test)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], color="b", marker="x")
    plt.show()

def random_forest_model(df, test_perc, target_var, m_depth, max_feat, num_feat = 10, show_importance = False, return_var = False):
    '''
    df: pd dataframe
    test_perc: percentage of testing sample
    target_var: target label
    m_depth: maximum depth of each decision tree
    max_feat: maximum of features used
    num_feat:number of features to keep based on importance
    show_importance: whether of print importance score
    return_var: whether to return most important features
    '''
    features = df.drop(columns = [target_var])
    feature_lst = features.columns
    X = np.array(features)
    y = np.array(df.loc[:,target_var])
    
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_perc, random_state=42)
    
    #fitting the tree
    rfc = RandomForestClassifier(n_estimators = 1000, random_state = 42, max_depth = m_depth, max_features = max_feat)
    rfc.fit(X_train, y_train)
    y_pred_train = rfc.predict(X_train)
    y_pred_test = rfc.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    print("Test accuracy:", accuracy)
    print('train accuracy', accuracy_score(y_train, y_pred_train),
      'test accuracy',  accuracy_score(y_test, y_pred_test))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    
    if show_importance:
        # Get numerical feature importances
        importances = list(rfc.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feat, round(importance, 2)) for feat, importance in zip(feature_lst, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    if return_var:
        return [x for x, y in feature_importances][:num_feat] + [target_var]
    
def GBM_model(dataset, target_var, test_perc, lrate, save_model = False):
    '''
    dataset: pd dataframe
    target_var: target label to predict
    test_perc: percentage of testing sample
    lrate: learning rate
    save_model: whether to save model
    '''
    
    X = dataset.drop(columns = target_var)
    y = dataset.loc[:, target_var]
    feature_lst = X.columns
    
    #establish the model
    model = GradientBoostingClassifier(learning_rate = lrate)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_perc, random_state=42)

    # fit the model on the training dataset
    model.fit(X_train, y_train)

    #prediction
    y_predicted_training = model.predict(X_train)
    y_predicted_test = model.predict(X_test)

    #getting the accuracy
    training_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('Training Score:', training_score)
    print('Testing Score:', test_score)
    print('Classication Report:') 
    print(classification_report(y_predicted_test, y_test))

    # Get numerical feature importances
    importances = list(model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feat, round(importance, 2)) for feat, importance in zip(feature_lst, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    var_lst = []
    for pair in feature_importances:
        if pair[1] > 0:
            print('Variable: {:20} Importance: {}'.format(*pair))
            var_lst.append(pair[0])

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], color="b", marker="x")
    plt.show()

    if save_model:
        filename = 'GBM_model.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model, var_lst
