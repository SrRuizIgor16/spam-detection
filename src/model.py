import pickle
import datetime
import os
from evaluation import Evaluation
from model_training import ModelTraining

class Model:
    def __init__(self, path, model=None):
        self.path = path
        self.model = model
        self.trained = False
        self.training_date = None
        self.train = None
        self.accuracy = None
        self.evaluation = None

    def __str__(self):
        return self.model.__class__.__name__

    def train_model(self, data, split=True, column=0, column_name='Cleaned_Message'):
        if self.model is not None:
            self.train = ModelTraining()
            X_train, X_test, y_train, y_test = self.train.pre_train(data, split=True, column_name=column_name)
            self.model.fit(X_train, y_train)
            self.training_date = datetime.datetime.now()
            self.trained = True
            self.accuracy = self.evaluate_model(y_test, self.model.predict(X_test))[1]
            print(f'Model trained successfully with accuracy: {self.accuracy * 100:.2f}%')
        else:
            raise ValueError('No model found')

    def save_model(self, name='model_instance.pkl'):
        path = self.path + name
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            print(f'Model instance saved successfully at {path}')
        except Exception as e:
            raise Exception(f'Failed to save model instance: {str(e)}')

    def predict(self, data):
        if self.model is not None:
            cln_data = self.train.clean_data(data)
            cln_vectorized = self.train.vectorize_data(cln_data)
            is_spam = self.model.predict(cln_vectorized)
            return is_spam
        else:
            raise ValueError('No model found')

    def evaluate(self):
        if self.model is None:
            raise ValueError('No model found')
        elif not self.trained or self.evaluation is None:
            self.evaluation = Evaluation(self.model)

    def evaluate_model(self, true_labels, predictions):
        if self.evaluation is not None:
            return self.evaluation.evaluate(true_labels, predictions)
        elif self.trained:
            self.evaluate()
            return self.evaluation.evaluate(true_labels, predictions)
        else:
            raise ValueError('Model not trained')

@staticmethod
def load_model(path):
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
    if os.path.exists(path):
        os.remove(path)
        print(f'Model instance at {path} removed successfully')
    else:
        raise FileNotFoundError(f'The file at {path} does not exist')

def main():
    pass

if __name__ == '__main__':
    main()
