from __future__ import division
import numpy as np
import math

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# c1 = [ 'dit is de eerste text', 'de is de tweede text nog wat info', 'dit is de derde text wat grappig dit henk'];
# c2 = [ 'dit is de tweede class text', 'jaja wat een uitstekende test', 'jaja dit is lastig zeg'];

# print start[1].split().count('de')

class BM25:
	def __init__(self, class1, class2):
		self.k1 = 1.5
		self.b = 0.75
		self.class1 = class1
		self.class2 = class2
		self.corpus = class1 + class2

		self.avg_dl = np.average([len(a.split()) for a in self.corpus])
		self.features = set([item for sublist in [a.split() for a in self.corpus] for item in sublist])

		self.N1 = len(self.class1)
		self.N2 = len(self.class2)

		# Build a cache of the IDF's to save time
		self.getidfs()
		# self.limitFeatures(1200)

	def df(self, term, clas):
		return len([a for a in clas if term in a])

	def vector(self, text):
		text = text.split()
		vector = []
		# enumerate the features
		for a in self.features:
			if a in text:
				# calculate the frequency of appearance
				tfi = text.count(a)
				k = self.k1 * ((1 - self.b) + self.b * (len(text)/self.avg_dl))
				tf = ((self.k1 + 1) * tfi) / (k + tfi)
				# get the IDF part from a cache, to save time
				idf = self.idf_dict[a]
				vector.append( tf * idf )
			else:
				vector.append( 0 )

		return vector

	def getidfs(self):
		self.idf_dict = {}
		for a in self.features:
			idf = 	((self.N1 - self.df(a, self.class1) + 0.5) * (self.df(a, self.class2) + 0.5)) / \
					((self.N2 - self.df(a, self.class2) + 0.5) * (self.df(a, self.class1) + 0.5))
			self.idf_dict[a] = math.log(idf)

	def limitFeatures(self, features):
		most_class1 = sorted(self.features, key=lambda x:self.idf_dict[x])[:int(features/2)]
		most_class2 = sorted(self.features, key=lambda x:self.idf_dict[x])[-int(features/2):]
		self.features = most_class1 + most_class2

	def vectorize(self, documents):
		matrix = []
		for text in documents:
			matrix.append( self.vector( text ))
		return matrix

# bm25 = BM25(c1,c2)
# training = bm25.vectorize(c1 + c2)

# model = RandomForestClassifier(n_estimators = 500)
# model = SVC(probability=1)
# model.fit( training, [1,1,1,0,0,0])
# print bm25.idf_dict
# print sorted(bm25.features, key=lambda x:bm25.idf_dict[x])
# print bm25.vector('henk')
# print model.predict_proba( bm25.vector('henk') ) 
