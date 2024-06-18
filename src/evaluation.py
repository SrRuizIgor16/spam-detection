import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
import os

import utils

class Evaluation():
    def __init__(self, model, path=None):
        self.model = model
        self.model_name = model.__class__.__name__
        self.conf_matrix = None
        self.accuracy = None
        self.path = self.set_path(path)

    def set_path(self, path):
        if path is None:
            path = f'../reports/{self.model_name}'
        self.path = path
        os.makedirs('../reports', exist_ok=True)
        os.makedirs(self.path, exist_ok=True)

    def evaluate(self, true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        self.conf_matrix = cm
        self.accuracy = accuracy
        return cm, accuracy

    # Confusion Matrix
    def plot_confusion_matrix(self, save_plots=False, show_plots=True):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.conf_matrix)
        disp.plot()
        disp.ax_.set_title('Confusion Matrix')
        if save_plots:
            utils.save_plot(plt, self.path, 'Confusion Matrix')
        if show_plots:
            plt.show()

    # ROC Curve
    def plot_roc_curve(self, y_test, y_score, save_plots=False, show_plots=True):

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

    # Precision-Recall Curve
    def plot_precision_recall_curve(self, y_test, y_score, save_plots=False, show_plots=True):
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

    # Learning Curve
    def plot_learning_curve(self, X, y, save_plots=False, show_plots=True):
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
            raise e
            # print(f'Error evaluating model: {str(e)}')
            return False



def compare_models(models, sorted=True, ascending=True, save_plots=False, path=None, view_plots=True):
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