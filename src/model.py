import pickle
import datetime
import os
import numpy as np
from evaluation import Evaluation
from model_training import ModelTraining

class Model:
    """
    A class representing a machine learning model for spam detection.

    Attributes
    ----------
    path : str
        Path to save or load the model.
    model : object
        The machine learning model instance.
    trained : bool
        Indicates if the model is trained.
    training_date : datetime
        Date when the model was trained.
    train : ModelTraining
        Instance of ModelTraining for data preprocessing.
    accuracy : float
        Accuracy of the model.
    evaluation : Evaluation
        Instance of Evaluation for model evaluation.
    vectorizer : CountVectorizer
        Vectorizer for transforming text data.
    """

    def __init__(self, path, model=None):
        """
        Initializes the Model class.

        Parameters
        ----------
        path : str
            Path to save or load the model.
        model : object, optional
            The machine learning model instance.
        """
        self.path = path
        self.model = model
        self.trained = False
        self.training_date = None
        self.train = None
        self.accuracy = None
        self.evaluation = None
        self.vectorizer = None

    def __str__(self):
        """Returns the name of the model class."""
        return self.model.__class__.__name__

    def train_model(self, data, split=True, column=0, column_name='Cleaned_Message'):
        """
        Trains the model using the provided data.

        Parameters
        ----------
        data : DataFrame
            The data to train the model on.
        split : bool
            Whether to split the data into training and testing sets.
        column : int
            The column index of the target variable.
        column_name : str
            The column name of the text data.
        """
        if self.model is not None:
            self.train = ModelTraining()
            X_train, X_test, y_train, y_test = self.train.pre_train(data, split=True, column_name=column_name)
            self.model.fit(X_train, y_train)
            self.training_date = datetime.datetime.now()
            self.trained = True
            self.accuracy = self.evaluate_model(y_test, self.model.predict(X_test))[1]
            self.vectorizer = self.train.cv
            print(f'Model trained successfully with accuracy: {self.accuracy * 100:.2f}%')
        else:
            raise ValueError('No model found')

    def save_model(self, name='model_instance.pkl'):
        """
        Saves the model to a file.

        Parameters
        ----------
        name : str
            The name of the file to save the model.
        """
        path = self.path + name
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            print(f'Model instance saved successfully at {path}')
        except Exception as e:
            raise Exception(f'Failed to save model instance: {str(e)}')

    def predict(self, data):
        """
        Makes a prediction using the model.

        Parameters
        ----------
        data : array-like
            The data to predict.

        Returns
        -------
        array-like
            The prediction results.
        """
        if self.model is not None:
            is_spam = self.model.predict(data)
            return is_spam
        else:
            raise ValueError('No model found')

    def evaluate(self):
        """
        Evaluates the model if it is trained.
        """
        if self.model is None:
            raise ValueError('No model found')
        elif not self.trained or self.evaluation is None:
            self.evaluation = Evaluation(self.model)

    def evaluate_model(self, true_labels, predictions):
        """
        Evaluates the model using true labels and predictions.

        Parameters
        ----------
        true_labels : array-like
            True labels.
        predictions : array-like
            Predicted labels by the model.

        Returns
        -------
        tuple
            Confusion matrix and accuracy.
        """
        if self.evaluation is not None:
            return self.evaluation.evaluate(true_labels, predictions)
        elif self.trained:
            self.evaluate()
            return self.evaluation.evaluate(true_labels, predictions)
        else:
            raise ValueError('Model not trained')

@staticmethod
def load_model(path):
    """
    Loads a model from a file.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
        -------
    Model
        The loaded model instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'The file at {path} does not exist')
    try:
        with open(path, 'rb') as f:
            model_instance = pickle.load(f)
            print(f'Model instance loaded successfully from {path}')
        return model_instance
    except Exception as e:
        raise Exception(f'Failed to load model instance: {str(e)}')

@staticmethod
def reset_model(path):
    """
    Deletes a model file.

    Parameters
    ----------
    path : str
        The path to the file to delete.
    """
    if os.path.exists(path):
        os.remove(path)
        print(f'Model instance at {path} removed successfully')
    else:
        raise FileNotFoundError(f'The file at {path} does not exist')

def main():
    pass

if __name__ == '__main__':
    main()