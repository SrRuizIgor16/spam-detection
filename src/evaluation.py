import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
import os

import utils

class Evaluation:
    """
    A class used to evaluate machine learning models and generate various evaluation plots.

    Attributes
    ----------
    model : object
        The machine learning model to evaluate.
    model_name : str
        The name of the model class.
    conf_matrix : array, shape (n_classes, n_classes)
        Confusion matrix.
    accuracy : float
        Accuracy of the model.
    path : str
        Directory where the plots will be saved.
    """

    def __init__(self, model, path=None):
        """
        Initializes the Evaluation class with the given model and sets up the path for saving plots.

        Parameters
        ----------
        model : object
            The machine learning model to evaluate.
        path : str, optional
            Directory where the plots will be saved.
        """
        self.model = model
        self.model_name = model.__class__.__name__
        self.conf_matrix = None
        self.accuracy = None
        self.path = self.set_path(path)

    def set_path(self, path):
        """
        Sets the path for saving plots. Creates directories if they do not exist.

        Parameters
        ----------
        path : str
            Directory where the plots will be saved.
        """
        if path is None:
            path = f'../reports/{self.model_name}'
        self.path = path
        os.makedirs('../reports', exist_ok=True)
        os.makedirs(self.path, exist_ok=True)

    def evaluate(self, true_labels, predictions):
        """
        Evaluates the model using true labels and predictions. Computes the confusion matrix and accuracy.

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
        cm = confusion_matrix(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        self.conf_matrix = cm
        self.accuracy = accuracy
        return cm, accuracy

    def plot_confusion_matrix(self, save_plots=False, show_plots=True):
        """
        Plots the confusion matrix.

        Parameters
        ----------
        save_plots : bool
            Whether to save the plot.
        show_plots : bool
            Whether to display the plot.
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=self.conf_matrix)
        disp.plot()
        disp.ax_.set_title('Confusion Matrix')
        if save_plots:
            utils.save_plot(plt, self.path, 'Confusion Matrix')
        if show_plots:
            plt.show()

    def plot_roc_curve(self, y_test, y_score, save_plots=False, show_plots=True):
        """
        Plots the ROC curve.

        Parameters
        ----------
        y_test : array-like
            True labels.
        y_score : array-like
            Predicted scores by the model.
        save_plots : bool
            Whether to save the plot.
        show_plots : bool
            Whether to display the plot.
        """
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if save_plots:
            utils.save_plot(plt, self.path, 'ROC Curve')
        if show_plots:
            plt.show()

    def plot_precision_recall_curve(self, y_test, y_score, save_plots=False, show_plots=True):
        """
        Plots the precision-recall curve.

        Parameters
        ----------
        y_test : array-like
            True labels.
        y_score : array-like
            Predicted scores by the model.
        save_plots : bool
            Whether to save the plot.
        show_plots : bool
            Whether to display the plot.
        """
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        plt.figure()
        plt.plot(recall, precision)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if save_plots:
            utils.save_plot(plt, self.path, 'Precision-Recall Curve')
        if show_plots:
            plt.show()

    def plot_learning_curve(self, X, y, save_plots=False, show_plots=True):
        """
        Plots the learning curve.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Training labels.
        save_plots : bool
            Whether to save the plot.
        show_plots : bool
            Whether to display the plot.
        """
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y)
        plt.figure()
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training error')
        plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation error')
        plt.title('Learning Curve')
        plt.xlabel('Training Set Size')
        plt.ylabel('Error')
        plt.legend()
        if save_plots:
            utils.save_plot(plt, self.path, 'Learning Curve')
        if show_plots:
            plt.show()

    def evaluation_metric(self, X, Y, save_plots=False, show_plots=True):
        """
        Evaluates the model using various metrics and plots.

        Parameters
        ----------
        X : array-like
            Features.
        Y : array-like
            Labels.
        save_plots : bool
            Whether to save the plots.
        show_plots : bool
            Whether to display the plots.

        Returns
        -------
        bool
            True if evaluation is successful, False otherwise.
        """
        try:
            if self.conf_matrix is None or self.accuracy is None:
                raise ValueError('No evaluation found')
            if self.path is None:
                self.set_path(None)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            y_score = self.model.predict(X_test)
            self.plot_confusion_matrix(save_plots=save_plots, show_plots=show_plots)
            self.plot_roc_curve(y_test, y_score, save_plots=save_plots, show_plots=show_plots)
            self.plot_precision_recall_curve(y_test, y_score, save_plots=save_plots, show_plots=show_plots)
            self.plot_learning_curve(X, Y, save_plots=save_plots, show_plots=show_plots)
            return True
        except Exception as e:
            print(f'Error evaluating model: {str(e)}')
            return False

def compare_models(models, sorted=True, ascending=True, save_plots=False, path=None, view_plots=True):
    """
    Compares multiple models based on their accuracy and generates a bar plot.

    Parameters
    ----------
    models : dict
        Dictionary containing model information.
    sorted : bool
        Whether to sort the models by accuracy.
    ascending : bool
        Whether to sort in ascending order.
    save_plots : bool
        Whether to save the plot.
    path : str
        Directory where the plot will be saved.
    view_plots : bool
        Whether to display the plot.
    """
    if sorted:
        models = utils.sort_models(models, ascending)
    plt.figure(figsize=(10, 8))
    for model_name, info in models.items():
        plt.bar(model_name, info['Accuracy']*100)
        plt.text(model_name, info['Accuracy']*100, f'{info["Accuracy"]*100:.2f}%', ha='center', va='bottom')
    plt.title('Model Accuracy Comparison Sorted by Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=10)
    if save_plots:
        if path is None:
            path = '../reports'
        utils.save_plot(plt, path, 'Model Accuracy Comparison')
    if view_plots:
        plt.show()

def main():
    pass

if __name__ == '__main__':
    main()