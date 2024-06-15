import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from model_training import ModelTraining

class Model:
	def __init__(self, config, path):
		self.config = config
		self.path = path
		self.model = None
		self.trained = False
		self.save = self.save_model(self.path)
		self.open = self.load_model(self.path)
		self.train = None

	def train_model(self, data, epochs, batch_size):
		if self.model is not None:
			self.train = ModelTraining(self.config, epochs=epochs, batch_size=batch_size)
			self.train.train(data)
			self.trained = True
		else:
			raise ValueError('No model found')

	def save_model(self, model):
		try:
			with open(model, 'wb') as f:
				pickle.dump(self.model, f)
		except Exception as e:
			raise Exception('No model to save'+ str(e))

	def load_model(self, model):
		try:
			with open(model, 'rb') as f:
				self.model = pickle.load(f)
		except Exception as e:
			raise FileNotFoundError('No model to load'+ str(e))

	def predict(self, data):
		if self.model is not None:
			cln_data = self.train.clean_data(data)
			isSpam = self.model.predict(cln_data)
			if isSpam == 1:
				print('Spam')
				return True
			else:
				print('Not Spam')
				return False
		else:
			raise ValueError('No model found')

	def evaluate(self, info, predictions):
		cm = confusion_matrix(info, predictions)
		accuracy = accuracy_score(info, predictions)
		print(f'Accuracy: {accuracy * 100:.2f} %')
		return cm, accuracy

	def summary(self):
		if self.trained:
			print('Model Summary')
			print('Model:', self.model)
			print('Training:', self.train)
			print('Trained:', self.trained)
			print('Accuracy:', self.train.accuracy)
		else:
			raise ValueError('No model found')
