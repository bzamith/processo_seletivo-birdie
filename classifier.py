from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

class Classifier:
	def __init__(self, save_name='post_indentifier.tsv',file_name='data_estag_ds.tsv',pre_process=True):
		self.count_vect = CountVectorizer()
		self.tf_transformer = TfidfTransformer()
		#data read from tsv file
		self.data_read, self.known_positive = self.read_dataset(file_name)
		#data_p is the preprocessed data (data we are going to predict)
		self.data_p = self.data_read[0:]
		if pre_process:
			self.data_p,self.positive_ex,self.negative_ex = self.process_dataset()
		#data_t is data_p after being transformeed (count vect and tf_transformer)
		self.data_t,self.data_train,self.data_test = self.split_dataset()
		#prediction with KMeans
		self.clf = self.clf_KMeans()
		self.prediction = self.clf.predict(self.data_t)
		self.prediction = self.convert_prediction()
		self.write_prediction(save_name,pre_process)
	#read dataset
	def read_dataset(self,file_name):
		known_positive = ''
		with open(file_name, 'r') as dataset_file:
			data = dataset_file.readlines()
			data = [item.strip().split('\t') for item in data]
			data_read = [item[1] for item in data[1:len(data)]]
		for i,offer in enumerate(data_read):
			data_read[i] = offer.lower()
			if 'smartphone' in data_read[i] and known_positive== '':
				known_positive = data_read[i]
		return data_read, known_positive
	#split dataset into train and test
	def split_dataset(self,test_proportion=0.2):
		data = self.count_vect.fit_transform(self.data_p)
		data = self.tf_transformer.fit_transform(data)
		data_train, data_test = train_test_split(data,test_size=test_proportion)
		return data,data_train,data_test
	#pre process the data, this means classifying likely instances based on the occurance of some words
	def process_dataset(self):
		positive = []
		positive_ex = []
		negative = []
		negative_ex = []
		data = self.data_p
		for i,offer in enumerate(data):
			if offer != self.known_positive:
				if 'smartphone' in offer:
					positive.append(i)
					positive_ex.append(offer)
				elif 'capa' in offer:
					negative.append(i)
					negative_ex.append(offer)
				elif 'tablet' in offer:
					negative.append(i)
					negative_ex.append(offer)
				elif 'pelicula' in offer or 'película' in offer:
					negative.append(i)
					negative_ex.append(offer)
				elif 'bumper' in offer or 'bumber' in offer:
					negative.append(i)
					negative_ex.append(offer)
				elif 'bracadeira' in offer:
					negative.append(i)
					negative_ex.append(offer)
				elif 'carregador' in offer:
					negative.append(i)
					negative_ex.append(offer)
		remove = positive + negative
		data = np.delete(data,remove)
		return data,positive_ex,negative_ex
	#KMeans, it can either pick the best k or use k=2
	def clf_KMeans(self,best_k=False,threshold=100):
		if best_k:
			score_anterior = KMeans(n_clusters=2).fit(self.data_train).score(self.data_test)
			diff_anterior = abs(score_anterior - KMeans(n_clusters=1).fit(self.data_train).score(self.data_test))
			for i in range(3,len(self.data_read)):
				km = KMeans(n_clusters=i).fit(self.data_train)
				score_atual = km.score(self.data_test)
				diff = abs(score_atual - score_anterior) 
				if abs(diff - diff_anterior) < threshold:
					break
				score_anterior = score_atual
				diff_anterior = diff
			nb_k=i
		else:
			nb_k = 2
		clf = KMeans(n_clusters=nb_k).fit(self.data_t)
		return clf
	#convert classes from K Means to "smartphone" or "not smartphone"
	def convert_prediction(self):
		try:
			if type(self.data_p).__module__ == np.__name__:
				index = self.data_p.tolist().index(self.known_positive)
			else:
				index = self.data_p.index(self.known_positive)
		except ValueError:
			print("Known positive not found in your dataset")	
		positive_class = self.prediction[index]
		new_prediction = []
		for item in self.prediction:
			if item == positive_class:
				new_prediction.append("smartphone")
			else:
				new_prediction.append("not_smartphone")
		return new_prediction
	#extract prediction
	def write_prediction(self,save_name,pre_process):
		with open(save_name, 'w') as f:
			for i,item in enumerate(self.prediction):
				f.write(self.data_p[i]+"\t")
				f.write(item+"\n")
			if pre_process:
				for item in self.positive_ex:
					f.write(item+"\tsmartphone\n")
				for item in self.negative_ex:
					f.write(item+"\tnot_smartphone\n")

teste = Classifier()
