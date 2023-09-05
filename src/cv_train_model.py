import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from src.utils.scorers import f1_w_threshold, acc_w_threshold, precision_w_threshold, recall_w_threshold


def cv_train_model(
        sklearn_model,
        X_train, 
        y_train, 
        model_file_name, 
        cv_scores_file_name, 
        kfolds=5,
        thresholds=np.arange(0.1, 1.0, 0.05),
        overwrite_current=False):
    # Check if the models and cv scores files exist inside the models and cv_model_scores directories, if not create them
    if (not os.path.exists(os.path.join('models', model_file_name)) and \
        not os.path.exists(os.path.join('cv_model_scores', cv_scores_file_name))) or \
        overwrite_current:
        
        print('Instantiating the data and model...\n')
        
        X_train = X_train.copy()
        y_train = y_train.copy()

        # Initialize the pipeline
        model = sklearn_model

        print('Starting the cross validation loops with different thresholds...\n')
        # Different threshold values to test
        thresholds = thresholds
        # Perform cross-validation with custom scoring function
        cv_scores_thresholds = []
        for threshold in thresholds:
            # Define custom scoring functions
            scoring_functions = {
                "F1-score": make_scorer(f1_w_threshold, needs_proba=True, threshold=threshold),
                "Accuracy": make_scorer(acc_w_threshold, needs_proba=True, threshold=threshold),
                "Precision": make_scorer(precision_w_threshold, needs_proba=True, threshold=threshold),
                "Recall": make_scorer(recall_w_threshold, needs_proba=True, threshold=threshold),
                "AUC": 'roc_auc',
                "Average Precision": 'average_precision'
            }
            print(f'Cross-validating with threshold = {threshold}...\n')
            # Perform cross-validation
            scores = cross_validate(model, X_train, y_train, cv=kfolds, scoring=scoring_functions)
            # Add a key to the dictionary with the threshold value
            scores['threshold'] = threshold
            cv_scores_thresholds.append(scores)

        # Train the model and saves it to a pickle file
        print('Training the model with the full data...\n')
        model.fit(X_train, y_train)
        print('Saving the model to a pickle file...\n')
        pickle.dump(model, open(os.path.join('models', model_file_name), 'wb'))

        # Creates a dataframe with the CV scores and saves them as a csv file
        cv_scores = pd.concat(
            [pd.DataFrame(cv_scores_thresholds[i]) 
             for i in range(len(cv_scores_thresholds))],
             ignore_index=True)
        
        print('Saving the CV scores to a csv file...\n')
        cv_scores.to_csv(os.path.join('cv_model_scores', cv_scores_file_name), index=False)

        print('The model has been trained and saved!\n')
        print('Returning the model and the CV scores...\n')
        return model, cv_scores

    else:
        print('This model has already been trained and saved!\n')
        print('Loading the model and the CV scores...\n')
        model = pickle.load(open(os.path.join('models', model_file_name), 'rb'))
        cv_scores = pd.read_csv(os.path.join('cv_model_scores', cv_scores_file_name))

        print('Returning the model and the CV scores...\n')
        return model, cv_scores
