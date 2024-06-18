import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from model import Model, load_model
from data_preprocessing import DataPreprocessing
from evaluation import compare_models

def save_cleaned_data(save=False):
    """
    Loads the data, preprocesses it, and saves the cleaned data to a CSV file.

    Returns
    -------
    DataPreprocessing
        The DataPreprocessing object.
    """
    dp = DataPreprocessing()
    if dp.data is not None:
        dp.preprocess_data(save=save)
        return dp
    else:
        print('Data not loaded. Cannot save cleaned data.')
        return None

def create_model(train_dp, model=None, path='../models/'):
    model_instance = Model(path=path, model=model)
    model_instance.train_model(train_dp.data, column_name='Message')
    model_instance.save_model(name=f'{model.__class__.__name__}.pkl')
    return model_instance

def get_model(model):
    path = f'../models/{model.__class__.__name__}.pkl'
    return load_model(path)

def obtain_models(key_models, dp=None, evaluate=True):
    models = {}
    for key_model in key_models:
        try:
            model = get_model(key_model)
        except FileNotFoundError:
            print(f'{key_model.__class__.__name__} model not found')
            if dp is None:
                dp = save_cleaned_data()
                if dp is None:
                    print('Failed to preprocess data. Exiting.')
                    return
            model = create_model(dp, model=key_model)
            print(f'{key_model.__class__.__name__} model created')

        info = {'Accuracy': model.accuracy,
                'Trained': model.trained,
                'Training Date': model.training_date}
        models[key_model.__class__.__name__] = info

        if evaluate:
            eval_model(model, dp)

    return models

def print_models(models):
    for model_name, info in models.items():
        print(f'{model_name}:')
        for key, value in info.items():
            print(f'{key}: {value}')

def eval_model(model, dp=None):
    if dp is None:
        dp = save_cleaned_data()
        if dp is None:
            print('Failed to preprocess data. Exiting.')
            return
    X, y = model.train.pre_train(dp.data, split=False, column_name='Message')
    print(f'Evaluating {model.__str__()}')
    model.evaluate()
    x = model.evaluation.evaluation_metric(X, y, save_plots=True, show_plots=False)
    if x:
        print(f'{model.__str__()} evaluation completed successfully')
    else:
        print(f'{model.__str__()} evaluation failed')


def main():
    key_models = [MultinomialNB(), GaussianNB(), SVC(), LogisticRegression(), RandomForestClassifier(),
                  DecisionTreeClassifier()]
    models = obtain_models(key_models, evaluate=False)
    compare_models(models, save_plots=True, view_plots=False)

if __name__ == '__main__':
    main()
