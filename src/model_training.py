import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class ModelTraining:
    """
    A class used for training machine learning models for spam detection.

    Attributes
    ----------
    pattern : str
        Regular expression pattern for text cleaning.
    repl : str
        Replacement string for text cleaning.
    stopwords : set
        Set of stopwords.
    max_features : int
        Maximum number of features for the vectorizer.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for shuffling the data.
    ps : PorterStemmer
        Porter stemmer for stemming words.
    cv : CountVectorizer
        Vectorizer for transforming text data.
    """

    download_stopwords = False

    def __init__(self, pattern='[^a-zA-Z]', repl=' ', stopwords=None, max_features=1500,
                 test_size=0.2, random_state=0):
        """
        Initializes the ModelTraining class.

        Parameters
        ----------
        pattern : str, optional
            Regular expression pattern for text cleaning.
        repl : str, optional
            Replacement string for text cleaning.
        stopwords : set, optional
            Set of stopwords.
        max_features : int, optional
            Maximum number of features for the vectorizer.
        test_size : float, optional
            Proportion of the dataset to include in the test split.
        random_state : int, optional
            Random seed for shuffling the data.
        """
        self.download_stopwords()
        self.pattern = pattern
        self.repl = repl
        self.stopwords = stopwords if stopwords is not None else set(nltk_stopwords.words('english'))
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.ps = PorterStemmer()
        self.cv = None

    def download_stopwords(self):
        """
        Downloads the NLTK stopwords if not already downloaded.
        """
        if not ModelTraining.download_stopwords:
            nltk.download('stopwords')
            ModelTraining.download_stopwords = True

    def clean_data(self, data, column_name='Cleaned_Message', pattern='[^a-zA-Z]', repl=' ', stopwords=None, ps=None):
        """
        Cleans the text data by removing non-alphabetic characters, converting to lowercase, and removing stopwords.

        Parameters
        ----------
        data : DataFrame
            The data to clean.
        column_name : str, optional
            The column name of the text data.
        pattern : str, optional
            Regular expression pattern for text cleaning.
        repl : str, optional
            Replacement string for text cleaning.
        stopwords : set, optional
            Set of stopwords.
        ps : PorterStemmer, optional
            Porter stemmer for stemming words.

        Returns
        -------
        list
            The cleaned text data.
        """
        corpus = []
        number = len(data)
        if stopwords is None:
            stopwords = self.stopwords
        if ps is None:
            ps = PorterStemmer()
        for i in range(number):
            if number > 1:
                review = re.sub(self.pattern, self.repl, data[column_name][i])
            else:
                review = re.sub(pattern, repl, data[0])
            review = review.lower()
            review = [ps.stem(word) for word in review.split() if word not in stopwords]
            review = ' '.join(review)
            corpus.append(review)
        return corpus

    def vectorize_data(self, corpus, max_features=None):
        """
        Vectorizes the text data using CountVectorizer.

        Parameters
        ----------
        corpus : list
            The text data to vectorize.
        max_features : int, optional
            Maximum number of features for the vectorizer.

        Returns
        -------
        array
            The vectorized text data.
        """
        if max_features is None:
            max_features = self.max_features
        self.cv = CountVectorizer(max_features=max_features)
        X = self.cv.fit_transform(corpus).toarray()
        return X

    def split_data(self, X, y):
        """
        Splits the data into training and testing sets.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        tuple
            Training and testing sets for features and labels.
        """
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def pre_train(self, data, split=True, column=0, column_name='Cleaned_Message'):
        """
        Prepares the data for training by cleaning and vectorizing it, and optionally splitting into training and testing sets.

        Parameters
        ----------
        data : DataFrame
            The data to prepare.
        split : bool, optional
            Whether to split the data into training and testing sets.
        column : int, optional
            The column index of the target variable.
        column_name : str, optional
            The column name of the text data.

        Returns
        -------
        tuple
            Prepared data for training and testing sets, or the entire dataset if split is False.
        """
        corpus = self.clean_data(data, column_name)
        X = self.vectorize_data(corpus)
        y = data.iloc[:, column].values
        if split:
            return self.split_data(X, y)
        else:
            return X, y

    def clean_predict(self, predict):
        """
        Cleans a single text message for prediction.

        Parameters
        ----------
        predict : str
            The text message to clean.

        Returns
        -------
        str
            The cleaned text message.
        """
        review = re.sub(pattern=self.pattern, repl=self.repl, string=predict)
        review = review.lower()
        review = [self.ps.stem(word) for word in review.split() if word not in self.stopwords]
        review = ' '.join(review)
        return review

def main():
    pass

if __name__ == '__main__':
    main()
