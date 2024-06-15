import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from model import Model

class ModelTraining(Model):
	def __init__(self, config, pattern='[^a-zA-Z]', repl=' ', stopwords=stopwords.words('english'), max_features=1500,
	             test_size=0.2, random_state=0, epochs=1, batch_size=10):
		Model.__init__(self, config, self.path)
		self.pattern = pattern
		self.repl = repl
		self.stopwords = stopwords
		self.max_features = max_features
		self.test_size = test_size
		self.random_state = random_state
		self.epochs = epochs
		self.batch_size = batch_size
		self.accuracy = None
		self.ps = PorterStemmer()

	def clean_data(self, data, column_name='Cleaned_Message'):
		corpus = []
		for i in range(0, len(data)):
			review = re.sub(pattern=self.pattern, repl=self.repl, string=data[column_name][i])
			review = review.lower()
			review = [self.ps.stem(word) for word in review.split() if not word in set(self.stopwords)]
			review = ' '.join(review)
			corpus.append(review)
		return corpus

	def vectorize_data(self, data, corpus=None, column=0, column_name='Cleaned_Message'):
		cv = CountVectorizer(max_features=self.max_features)
		if not corpus:
			corpus = self.clean_data(data, column_name=column_name)
		X = cv.fit_transform(corpus).toarray()
		y = data.iloc[:, column].values
		return X, y

	def split_data(self, X, y):
		return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

	def pre_train(self, data, corpus=None, column=0,):
		X, y = self.vectorize_data(data, corpus, column)
		X_train, X_test, y_train, y_test = self.split_data(X, y)
		return X_train, X_test, y_train, y_test

	def train(self, data, corpus=None, column=0):
		X_train, X_test, y_train, y_test = self.pre_train(data, corpus, column)
		self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
		cm, acc = self.evaluate(X_test, y_test)
		self.accuracy = acc
		return cm, acc

	def clean_predict(self, predict):
		review = re.sub(pattern=self.pattern, repl=self.repl, string=predict)
		review = review.lower()
		review = [self.ps.stem(word) for word in review.split() if not word in set(self.stopwords)]
		review = ' '.join(review)
		return review
