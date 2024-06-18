from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import utils
from evaluation import compare_models

def prediction(model, to_predict):
    """
    Makes a prediction using the specified model.

    Parameters
    ----------
    model : Model
        The trained model to use for prediction.
    to_predict : str
        The text message to predict.

    Returns
    -------
    int
        The prediction result (1 for spam, 0 for not spam).
    """
    model_instance = model
    cleaned_data = model_instance.train.clean_predict(to_predict)
    vectorized_data = model_instance.vectorizer.transform([cleaned_data]).toarray()
    prediction = model_instance.predict(vectorized_data)
    return prediction[0]

def print_text(models):
    """
    Prints a list of models to choose from.

    Parameters
    ----------
    models : list
        List of model names.
    """
    text = '''Select the number of the model to predict:\n--------------------------------'''
    for i, model in enumerate(models):
        text += f'\n\t {i + 1}- {model}'
    text += f'\n\t {len(models) + 1}- Completed\n'
    text += '\n--------------------------------\n'
    print(text)

def select_model(models_info, models):
    """
    Selects a model for prediction.

    Parameters
    ----------
    models_info : dict
        Dictionary containing model information.
    models : dict
        Dictionary containing model instances.

    Returns
    -------
    int
        The prediction result (1 for spam, 0 for not spam).
    """
    print_text(models_info)
    model = int(input()) - 1

    if model < 0 or model > len(models_info):
        print('Invalid model. Please try again.')
        return select_model(models_info, models)
    elif model == len(models_info):
        print('Enter a message to predict: ')
        message = input().strip()
        predictions = 0
        total_accuracy = sum([model.accuracy for model in models.values()])

        for model_name in list(models_info.keys()):
            model = models[model_name]
            a = prediction(model, message)
            predictions += a * (model.accuracy / total_accuracy)

        if predictions >= 0.49:
            return 1
        else:
            return 0
    else:
        model_name = list(models_info.keys())[model]
        model = models[model_name]
        print(f'You selected {model}')
        print('Enter a message to predict: ')
        message = input().strip()
        print(f'Predicting with {model}...')
        return prediction(model, message)

def set_up():
    """
    Sets up the models for prediction.

    Returns
    -------
    tuple
        Dictionary of model information and dictionary of model instances.
    """
    key_models = [MultinomialNB(), GaussianNB(), SVC(), LogisticRegression(), RandomForestClassifier(),
                  DecisionTreeClassifier()]
    models_info, models = utils.obtain_models(key_models, evaluate=False)
    compare_models(models_info, save_plots=True, view_plots=False)
    return models_info, models

def main():
    """
    Main function to run the prediction program.
    """
    models_info, models = set_up()
    print('Models set up.\n')
    next = True
    while next:
        print('Starting prediction...')
        if select_model(models_info, models) == 1:
            prediction = 'spam'
        else:
            prediction = 'not spam'
        print(f'The prediction is: {prediction}\n\nThank you for using this program! :')
        print('Do you want to predict another message? (y/n)')
        if input().lower() != 'y':
            next = False

if __name__ == '__main__':
    main()