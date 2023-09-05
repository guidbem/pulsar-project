import os
import pickle
import optuna
import numpy as np
from src.cv_train_model import cv_train_model


def tune_hyperparams(
        model,
        X_train,
        y_train,
        param_grid,
        scoring_metric,
        n_trials,
        cv=5,
        n_jobs=1,
        random_state=42,
        optuna_obj_filename='optuna_search_obj.pkl',
        trials_scores_filename='trials_scores.csv',
        overwrite_current=False):
    """Performs a hyperparameter search using Optuna.

    Params:
        model: A sklearn model object.
        X_train: A pandas DataFrame containing the training data.
        y_train: A pandas Series containing the training labels.
        param_distribs: A dictionary containing the hyperparameter names and distributions.
        scoring_metric: A string containing the scoring metric to optimize.
        cv: A sklearn cross-validation object.
        n_trials: An integer containing the number of trials to perform.
        n_jobs: An integer containing the number of jobs to run in parallel.
        random_state: An integer containing the random state to use.

    Returns:
        An OptunaSearchCV object.
    """
    # Check if the models and cv scores files exist inside the models and cv_model_scores directories, if not create them
    if (not os.path.exists(os.path.join('models', optuna_obj_filename)) and \
        not os.path.exists(os.path.join('cv_model_scores', trials_scores_filename))) or \
        overwrite_current:
    
        # Initialize the Hyperparameter search object
        optuna_search = optuna.integration.OptunaSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring_metric,
            cv=cv,
            n_trials=n_trials,
            n_jobs=n_jobs,
            random_state=random_state)
        
        print('Initialized the hyperparameter search with the following attributes:\n')
        print(optuna_search)
        
        # Perform the hyperparameter search
        print('Starting the hyperparameter search...\n')
        optuna_search.fit(X_train, y_train)

        # Print the best hyperparameters
        print(f'Best hyperparameters: {optuna_search.best_params_}')
        print(f'Best {scoring_metric} score: {optuna_search.best_score_}')

        # Saves the trials scores to a csv file
        print('Saving the trials scores to a csv file...\n')
        trials_df = optuna_search.trials_dataframe()
        trials_df.to_csv(os.path.join('cv_model_scores', trials_scores_filename), index=False)

        # Saves the optuna search object to a pickle file
        print('Saving the optuna search object to a pickle file...\n')
        pickle.dump(optuna_search, open(os.path.join('models', optuna_obj_filename), 'wb'))

        return optuna_search
    else:
        print('This hyperparameter search has already been conducted!\n')
        print('Loading the optuna search object...\n')
        optuna_search = pickle.load(open(os.path.join('models', optuna_obj_filename), 'rb'))

        print('Returning the search object...\n')
        return optuna_search


def cv_train_best_model(
        optuna_search_obj,
        model,
        X_train,
        y_train,
        model_file_name,
        cv_scores_file_name,
        kfolds=5,
        thresholds=np.arange(0.1, 1.0, 0.05),
        overwrite_current=False):
    """Performs cross-validation with different thresholds and 
        the best hyperparameters found by Optuna.

    Params:
        optuna_search_obj: An OptunaSearchCV object.
        X_train: A pandas DataFrame containing the training data.
        y_train: A pandas Series containing the training labels.
        model_file_name: A string containing the name of the file to save the model.
        cv_scores_file_name: A string containing the name of the file to save the CV scores.
        kfolds: An integer containing the number of folds to use in cross-validation.
        thresholds: A numpy array containing the thresholds to use in cross-validation.
        overwrite_current: A boolean indicating whether to overwrite the current model and CV scores.

    Returns:
        A dictionary containing the CV scores.
    """
    # Get the best hyperparameters
    best_params = optuna_search_obj.best_params_

    # Set the best hyperparameters in the model
    model.set_params(**best_params)

    # Train the model with cross-validation
    trained_model, cv_scores = cv_train_model(
        model,
        X_train,
        y_train,
        model_file_name,
        cv_scores_file_name,
        kfolds=kfolds,
        thresholds=thresholds,
        overwrite_current=overwrite_current)
    
    return trained_model, cv_scores