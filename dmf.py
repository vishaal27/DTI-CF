import numpy as np
import json
from pprint import pprint
import re
import pandas as pd
import tensorflow as tf
import argparse
import sys
import os
import heapq
import math
import pickle
import matplotlib.pyplot as plt

DATA_PATH="data/"
RATINGS_MATRIX_FILE='gpcr.pickle'
USER_FILE=DATA_PATH+"users.csv"
MOVIES_FILE=DATA_PATH+"movies.csv"
RATINGS_FILE=DATA_PATH+"ratings_correctFormat.json"
TRAIN_FOLDS=["fold1_train.json", "fold2_train.json", "fold3_train.json", "fold4_train.json", "fold5_train.json"]
TEST_FOLDS=["fold1_test.json", "fold2_test.json", "fold3_test.json", "fold4_test.json", "fold5_test.json"]

TRAIN_FOLDS_FILES=[]
TEST_FOLDS_FILES=[]

ITERATIONS=[]
LOSSES=[]
NDCGS=[]
ITR=0

ITRS=[]
ITRR=0

for i in TRAIN_FOLDS:
	TRAIN_FOLDS_FILES.append(DATA_PATH+i)

for i in TEST_FOLDS:
	TEST_FOLDS_FILES.append(DATA_PATH+i)

class Model:
	def __init__(self, args, rating_matrix, ratings):
		self.dataName = 'FlickscoreData-26oct2018'
		# self.dataSet = DataSet(self.dataName)
		self.shape = np.asarray(rating_matrix).shape
		self.maxRate = 1.0

		# self.train = self.dataSet.train
		# self.test = self.dataSet.test

		self.rating_matrix=rating_matrix
		# self.users=users
		# self.movies=movies
		# self.movie_ids=movie_ids
		# self.user_ids=user_ids
		self.ratings=ratings

		self.train, self.test=self.train_test_split(test_size=0.2)

		print(self.train.shape, self.test.shape)

		self.trainDict=self.getTrainDict()

		self.negNum = 7
		self.testNeg = self.getTestNeg(self.test, 99)
		self.add_embedding_matrix()

		self.add_placeholders()

		self.userLayer = args.userLayer
		self.itemLayer = args.itemLayer
		self.add_model()

		self.add_loss()

		self.lr = args.lr
		self.add_train_step()

		self.checkPoint = args.checkPoint
		self.init_sess()

		self.maxEpochs = args.maxEpochs
		self.batchSize = args.batchSize

		self.topK = args.topK
		self.earlyStop = args.earlyStop

	def train_test_split(self, test_size):
		train=[]
		test=[]

		test_len=test_size*len(self.ratings)

		for i in range(int(test_len)):
			test.append(self.ratings[i])

		for i in range(int(test_len), len(self.ratings)):
			train.append(self.ratings[i])

		return np.asarray(train), np.asarray(test)

	def getTestNeg(self, testData, negNum):
		user = []
		item = []
		# print('test'+str(len(testData)))
		for s in testData:
			tmp_user = []
			tmp_item = []
			u = s[0]
			i = s[1]
			tmp_user.append(u)
			tmp_item.append(i)
			neglist = set()
			neglist.add(i)
			# print('Negnum'+str(negNum))
			for t in range(negNum):
				j = np.random.randint(self.shape[1])

				it=0
				while ((u, j) in self.trainDict or j in neglist) and it<100:
					# print(j, it)
					it+=1
					j = np.random.randint(self.shape[1])
				neglist.add(j)
				tmp_user.append(u)
				tmp_item.append(j)
			user.append(tmp_user)
			item.append(tmp_item)
		return [np.array(user), np.array(item)]

	def getTrainDict(self):
			dataDict = {}
			for i in self.train:
				dataDict[(i[0], i[1])] = i[2]
			return dataDict

	def add_placeholders(self):
		self.user = tf.placeholder(tf.int64)
		self.item = tf.placeholder(tf.int64)
		self.rate = tf.placeholder(tf.float64)
		self.drop = tf.placeholder(tf.float64)

	def add_embedding_matrix(self):
		# print(type(self.rating_matrix[0][2]))
		self.user_item_embedding = tf.convert_to_tensor(self.rating_matrix, dtype=tf.float64)
		self.item_user_embedding = tf.transpose(self.user_item_embedding)

	def add_model(self):
		user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
		item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

		def init_variable(shape, name):
			return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float64, stddev=0.01), name=name)

		with tf.name_scope("User_Layer"):
			user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
			print(type(user_input), type(tf.convert_to_tensor(user_W1, dtype=tf.float64)))
			user_out = tf.matmul(tf.convert_to_tensor(user_input, dtype=tf.float64), tf.convert_to_tensor(user_W1))
			for i in range(0, len(self.userLayer)-1):
				W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
				b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
				user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

		with tf.name_scope("Item_Layer"):
			item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
			item_out = tf.matmul(item_input, item_W1)
			for i in range(0, len(self.itemLayer)-1):
				W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
				b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
				item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

		norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
		norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
		self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keepdims=False) / (norm_item_output* norm_user_output)
		self.y_ = tf.maximum(np.float64(1e-6), self.y_)

	def add_loss(self):
		regRate = self.rate / self.maxRate
		losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
		loss = -tf.reduce_sum(losses)
		# regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
		# self.loss = loss + self.reg * regLoss
		self.loss = loss

	def add_train_step(self):
		'''
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.lr = tf.train.exponential_decay(self.lr, global_step,
											 self.decay_steps, self.decay_rate, staircase=True)
		'''
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_step = optimizer.minimize(self.loss)

	def init_sess(self):
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.config.allow_soft_placement = True
		self.sess = tf.Session(config=self.config)
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		if os.path.exists(self.checkPoint):
			[os.remove(f) for f in os.listdir(self.checkPoint)]
		else:
			os.mkdir(self.checkPoint)

	def getInstances(self, data, negNum):
		user = []
		item = []
		rate = []
		for i in data:
			user.append(i[0])
			item.append(i[1])
			rate.append(i[2])
			# print(i)
			for t in range(negNum):
				j = np.random.randint(self.shape[1])
				it=0
				while (i[0], j) in self.trainDict and it<100:
					j = np.random.randint(self.shape[1])
					it+=1
				user.append(i[0])
				item.append(j)
				rate.append(0.0)
		return np.array(user), np.array(item), np.array(rate)

	def run(self):
		global ITRR
		global NDCGS
		global ITRS
		global ITERATIONS
		global LOSSES

		best_hr = -1
		best_NDCG = -1
		best_epoch = -1
		print("Start Training!")
		for epoch in range(self.maxEpochs):
			print("="*20+"Epoch ", epoch, "="*20)
			# ITERATIONS.append(epoch)
			self.run_epoch(self.sess)
			print('='*50)
			print("Start Evaluation!")
			hr, NDCG = self.evaluate(self.sess, self.topK)
			ITRS.append(ITRR)
			ITRR+=1
			NDCGS.append(NDCG)
			print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
			if hr > best_hr or NDCG > best_NDCG:
				best_hr = hr
				best_NDCG = NDCG
				best_epoch = epoch
				self.saver.save(self.sess, self.checkPoint)
			if epoch - best_epoch > self.earlyStop:
				print("Normal Early stop!")
				break
			print("="*20+"Epoch ", epoch, "End"+"="*20)


		plt.plot(ITERATIONS, LOSSES)
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.show()

		plt.plot(ITRS, NDCGS)
		plt.xlabel('Epochs')
		plt.ylabel('NDCG')
		plt.show()


		print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
		print("Training complete!")

	def run_epoch(self, sess, verbose=10):

		global ITRR
		global NDCGS
		global ITRS
		global ITERATIONS
		global LOSSES
		global ITR

		train_u, train_i, train_r = self.getInstances(self.train, self.negNum)
		train_len = len(train_u)
		shuffled_idx = np.random.permutation(np.arange(train_len))
		train_u = train_u[shuffled_idx]
		train_i = train_i[shuffled_idx]
		train_r = train_r[shuffled_idx]

		num_batches = len(train_u) // self.batchSize + 1

		losses = []
		for i in range(num_batches):
			min_idx = i * self.batchSize
			max_idx = np.min([train_len, (i+1)*self.batchSize])
			train_u_batch = train_u[min_idx: max_idx]
			train_i_batch = train_i[min_idx: max_idx]
			train_r_batch = train_r[min_idx: max_idx]

			feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
			_, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
			losses.append(tmp_loss)


			if verbose and i % verbose == 0:
				sys.stdout.write('\r{} / {} : loss = {}'.format(
					i, num_batches, np.mean(losses[-verbose:])
				))
				sys.stdout.flush()
		loss = np.mean(losses)

		LOSSES.append(tmp_loss)
		ITERATIONS.append(ITR)
		ITR+=1

		print("\nMean loss in this epoch is: {}".format(loss))
		return loss

	def create_feed_dict(self, u, i, r=None, drop=None):
		return {self.user: u,
				self.item: i,
				self.rate: r,
				self.drop: drop}

	def evaluate(self, sess, topK):
		def getHitRatio(ranklist, targetItem):
			for item in ranklist:
				if item == targetItem:
					return 1
			return 0
		def getNDCG(ranklist, targetItem):
			for i in range(len(ranklist)):
				item = ranklist[i]
				if item == targetItem:
					return math.log(2) / math.log(i+2)
			return 0


		hr =[]
		NDCG = []
		testUser = self.testNeg[0]
		testItem = self.testNeg[1]
		for i in range(len(testUser)):
			target = testItem[i][0]
			feed_dict = self.create_feed_dict(testUser[i], testItem[i])
			predict = sess.run(self.y_, feed_dict=feed_dict)

			item_score_dict = {}

			for j in range(len(testItem[i])):
				item = testItem[i][j]
				item_score_dict[item] = predict[j]

			ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

			tmp_hr = getHitRatio(ranklist, target)
			tmp_NDCG = getNDCG(ranklist, target)
			hr.append(tmp_hr)
			NDCG.append(tmp_NDCG)
		return np.mean(hr), np.mean(NDCG)

class User():
	
	def __init__(self, name, languages, occupation, address, dob, gender):
		self.name=name
		self.languages=languages
		self.occupation=occupation
		self.address=address
		self.dob=dob
		self.gender=gender

	def display(self):
		print(self.name, self.languages, self.occupation, self.address, self.dob, self.gender)

class Movie():

	def __init__(self, movie_id, description, language, released, rating, writer, director, cast, genre, name):
		self.movie_id=movie_id
		self.description=description
		self.language=language
		self.released=released
		self.rating=rating
		self.writer=writer
		self.director=director
		self.cast=cast
		self.genre=genre
		self.name=name

	def display(self):
		print(self.movie_id, self.description, self.language, self.released, self.rating, self.writer, self.director, self.cast, self.genre, self.name)
		
def find_between(s, start, end):
  index1=s.find(start)
  index2=s.find(end, index1+1)
  return (s[index1+2:index2])

def retrieve_users():

	user_list=[]
	with open(USER_FILE, 'rb') as f:
		data=f.readlines()
		for d in data[1:]:
			l=d.decode('utf8').rstrip('\n').split(",")
			n=len(l)
			name=l[0][1:len(l[0])-1]
			occupation=l[n-4][1:len(l[n-4])-1]
			address=l[n-3][1:len(l[n-3])-1]
			dob=l[n-2][1:len(l[n-2])-1]
			gender=l[n-1][1:len(l[n-1])-1]
			
			st='""'
			lang_list=[]

			for i in range(1,n-4):
				result=find_between(l[i], st, st)
				lang_list.append(result)

			user_list.append(User(name, lang_list, occupation, address, dob, gender))
	return user_list

def retrieve_movies():
	movies_list=[]
	data_frame=pd.read_csv(MOVIES_FILE)
	movies_list=data_frame.values
	return movies_list

def define_movie_id(movies):
	movie_ids={}

	for i in range(len(movies)):
		movie_ids[movies[i][0]]=i

	return movie_ids

def define_user_id(users):
	user_ids={}
	user_names={}

	for i in range(len(users)):
		user_ids[users[i].name]=i
	
	for i in range(len(users)):
		user_names[i]=users[i].name

	return user_ids, user_names

def retrieve_ratings_matrix(movies, users, movie_ids, user_ids, user_names):
	ratings_list=[]
	data_frame=pd.read_json(RATINGS_FILE)
	ratings_list=data_frame.values
	
	rated_users=[]
	ratings_of_users=[]

	for i in range(len(ratings_list)):
		rated_users.append(ratings_list[i][0])
		ratings_of_users.append(ratings_list[i][1])

	user_item_matrix=[]
	ratings=[]

	for i in range(len(users)):
		l=[]
		for j in range(len(movies)):
			l.append(0)
		user_item_matrix.append(l)

	for i in range(len(rated_users)):
		user_id=user_ids[rated_users[i]]
		for key in list(ratings_of_users[i].keys()):
			if('submit' not in key):
				item_id=movie_ids[key]
				user_item_matrix[user_id][item_id]=float(ratings_of_users[i][key][0])
				li=[]
				li.append(float(user_id))
				li.append(float(item_id))
				li.append(float(ratings_of_users[i][key][0]))
				ratings.append(li)


def ratings_for_project(RATINGS_MATRIX):

	ratings=[]

	for i in range(len(RATINGS_MATRIX)):
		for j in range(len(RATINGS_MATRIX[i])):
			li=[]
			li.append(i)
			li.append(j)
			li.append(RATINGS_MATRIX[i][j])
			ratings.append(li)

	return ratings


	# for key1 in list(user_ids.keys()):
	# 	print(user_ids[key1])
	# 	print(ratings_of_users[0])
	# 	print(key1)
	# 	print(len(user_ids))
	# 	# print('thugg', str(ratings_of_users[user_ids[key1]]))
	# 	for key2 in list(ratings_of_users[user_ids[key1]].keys()):
	# 		# print(ratings_of_users[i])
	# 		if('submit' not in key2):
	# 			user_item_matrix[user_ids[key1]][movie_ids[key2]]=float(ratings_of_users[user_ids[key1]][key2][0])
	# 			li=[]
	# 			li.append(float(user_ids[key1]))
	# 			li.append(float(movie_ids[key2]))
	# 			li.append(float(ratings_of_users[user_ids[key1]][key2][0]))
	# 			# print(i, movie_ids[key], float(ratings_of_users[i][key][0]))
	# 			ratings.append(li)

	return np.asarray(user_item_matrix), np.asarray(ratings)

# def run_dmf():



# USERS=retrieve_users()
# MOVIES=retrieve_movies()
# MOVIE_IDS=define_movie_id(MOVIES)
# USER_IDS, USER_NAMES=define_user_id(USERS)
RATINGS_MATRIX=pickle.load(open(RATINGS_MATRIX_FILE, 'rb'))
RATINGS=ratings_for_project(RATINGS_MATRIX)
# RATINGS_MATRIX, RATINGS=retrieve_ratings_matrix(MOVIES, USERS, MOVIE_IDS, USER_IDS, USER_NAMES)
# print(RATINGS_MATRIX.shape, RATINGS.shape)

parser=argparse.ArgumentParser(description="Options")

parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
# parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
parser.add_argument('-topK', action='store', dest='topK', default=10)

args = parser.parse_args()
classifier = Model(args, RATINGS_MATRIX, RATINGS)

# print("Arguments:", args)
classifier.run()


