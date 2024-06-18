import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class ModelTraining:
    download_stopwords = False

    def __init__(self, pattern='[^a-zA-Z]', repl=' ', stopwords=None, max_features=1500,
                 test_size=0.2, random_state=0):
        self.download_stopwords()
        self.pattern = pattern
        self.repl = repl
        self.stopwords = stopwords if stopwords is not None else set(nltk_stopwords.words('english'))
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.ps = PorterStemmer()


    def download_stopwords(self):
        if not ModelTraining.download_stopwords:
            nltk.download('stopwords')
            ModelTraining.download_stopwords = True

    def clean_data(self, data, column_name='Cleaned_Message'):
        corpus = []
        for i in range(len(data)):
            review = re.sub(self.pattern, self.repl, data[column_name][i])
            review = review.lower()
            review = [self.ps.stem(word) for word in review.split() if word not in self.stopwords]
            review = ' '.join(review)
            corpus.append(review)
        return corpus

    def vectorize_data(self, corpus):
        cv = CountVectorizer(max_features=self.max_features)
        X = cv.fit_transform(corpus).toarray()
        return X

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def pre_train(self, data, split=True, column=0, column_name='Cleaned_Message'):
        corpus = self.clean_data(data, column_name)
        X = self.vectorize_data(corpus)
        y = data.iloc[:, column].values
        if split:
            return self.split_data(X, y)
        else:
            return X, y

    def clean_predict(self, predict):
        review = re.sub(pattern=self.pattern, repl=self.repl, string=predict)
        review = review.lower()
        review = [self.ps.stem(word) for word in review.split() if word not in self.stopwords]
        review = ' '.join(review)
        return review

def main():
    pass

if __name__ == '__main__':
    main()
