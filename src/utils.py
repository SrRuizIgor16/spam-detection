import os
from model import Model, load_model
from data_preprocessing import DataPreprocessing

def sort_models(models, ascending=True):
    """
    Sorts the models based on their accuracy.

    Parameters
    ----------
    models : dict
        Dictionary containing model information.
    ascending : bool
        Whether to sort in ascending order.

    Returns
    -------
    dict
        Sorted dictionary of models.
    """
    return {k: v for k, v in sorted(models.items(), key=lambda item: item[1]['Accuracy'], reverse=ascending)}

def save_plot(plt, path, name='plot'):
    """
    Saves the plot to a file.

    Parameters
    ----------
    plt : object
        Matplotlib plot object.
    path : str
        Directory where the plot will be saved.
    name : str
        The name of the plot file.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path += f'/{name}.png'
    try:
        plt.savefig(path)
        print(f'Plot "{name}" saved successfully in {path} folder.')
    except Exception as e:
        print(f'Error saving plot:', e)
    plt.close()

def save_cleaned_data(save=False):
    """
    Loads the data, preprocesses it, and saves the cleaned data to a CSV file.

    Parameters
    ----------
    save : bool
        Whether to save the cleaned data to a CSV file.

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
    """
    Creates and trains a model, then saves it to a file.

    Parameters
    ----------
    train_dp : DataPreprocessing
        The DataPreprocessing object containing the training data.
    model : object, optional
        The machine learning model to train.
    path : str, optional
        Directory where the model will be saved.

    Returns
    -------
    Model
        The trained model instance.
    """
    model_instance = Model(path=path, model=model)
    model_instance.train_model(train_dp.data, column_name='Message')
    model_instance.save_model(name=f'{model.__class__.__name__}.pkl')
    return model_instance

def get_model(model):
    """
    Loads a model from a file.

    Parameters
    ----------
    model : object
        The machine learning model class.

    Returns
    -------
    Model
        The loaded model instance.
    """
    path = f'../models/{model.__class__.__name__}.pkl'
    return load_model(path)

def obtain_models(key_models, dp=None, evaluate=True):
    """
    Obtains and optionally evaluates multiple models.

    Parameters
    ----------
    key_models : list
        List of machine learning model instances.
    dp : DataPreprocessing, optional
        The DataPreprocessing object containing the training data.
    evaluate : bool, optional
        Whether to evaluate the models.

    Returns
    -------
    tuple
        Dictionary of model information and dictionary of model instances.
    """
    models = {}
    models_info = {}
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

        if evaluate:
            eval_model(model, dp)

        info = {'Accuracy': model.accuracy,
                'Trained': model.trained,
                'Training Date': model.training_date}

        models_info[key_model.__class__.__name__] = info
        models[key_model.__class__.__name__] = model

    return models_info, models

def print_models(models):
    """
    Prints information about the models.

    Parameters
    ----------
    models : dict
        Dictionary containing model information.
    """
    for model_name, info in models.items():
        print(f'{model_name}:')
        for key, value in info.items():
            print(f'{key}: {value}')

def eval_model(model, dp=None):
    """
    Evaluates a model using the provided data.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    dp : DataPreprocessing, optional
        The DataPreprocessing object containing the data.
    """
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
