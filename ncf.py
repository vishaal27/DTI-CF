import numpy as np
import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from time import time
import sys
import GMF, MLP
import argparse
import scipy.sparse as sp
import pandas as pd 
import pickle
import matplotlib.pyplot as plt

DATA_PATH="data/"
USER_FILE=DATA_PATH+"users.csv"
MOVIES_FILE=DATA_PATH+"movies.csv"
RATINGS_FILE=DATA_PATH+"ratings_correctFormat.json"
TRAIN_FOLDS=["fold1_train.json", "fold2_train.json", "fold3_train.json", "fold4_train.json", "fold5_train.json"]
TEST_FOLDS=["fold1_test.json", "fold2_test.json", "fold3_test.json", "fold4_test.json", "fold5_test.json"]
RATINGS_MATRIX_FILE='e.pickle'

TRAIN_FOLDS_FILES=[]
TEST_FOLDS_FILES=[]

for i in TRAIN_FOLDS:
	TRAIN_FOLDS_FILES.append(DATA_PATH+i)

for i in TEST_FOLDS:
	TEST_FOLDS_FILES.append(DATA_PATH+i)

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


#################### Arguments ####################
def parse_args():
	parser = argparse.ArgumentParser(description="Run NeuMF.")
	# parser.add_argument('--path', nargs='?', default='Data/',
	#                     help='Input data path.')
	# parser.add_argument('--dataset', nargs='?', default='ml-1m',
	#                     help='Choose a dataset.')
	parser.add_argument('--epochs', type=int, default=100,
						help='Number of epochs.')
	parser.add_argument('--batch_size', type=int, default=256,
						help='Batch size.')
	parser.add_argument('--num_factors', type=int, default=8,
						help='Embedding size of MF model.')
	parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
						help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
	parser.add_argument('--reg_mf', type=float, default=0,
						help='Regularization for MF embeddings.')                    
	parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
						help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
	parser.add_argument('--num_neg', type=int, default=4,
						help='Number of negative instances to pair with a positive instance.')
	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate.')
	parser.add_argument('--learner', nargs='?', default='adam',
						help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
	parser.add_argument('--verbose', type=int, default=1,
						help='Show performance per X iterations')
	parser.add_argument('--out', type=int, default=1,
						help='Whether to save the trained model.')
	parser.add_argument('--mf_pretrain', nargs='?', default='',
						help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
	parser.add_argument('--mlp_pretrain', nargs='?', default='',
						help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
	return parser.parse_args()

def init_normal(shape, name=None):
	return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
	assert len(layers) == len(reg_layers)
	num_layer = len(layers) #Number of layers in the MLP
	# Input variables
	user_input = Input(shape=(1,), dtype='float64', name = 'user_input')
	item_input = Input(shape=(1,), dtype='float64', name = 'item_input')
	
	# Embedding layer
	MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
								  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)
	MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
								  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)   

	MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]//2), name = "mlp_embedding_user",
								  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
	MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]//2), name = 'mlp_embedding_item',
								  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   

	print(type(user_input))
	
	# MF part
	mf_user_latent = Flatten()(MF_Embedding_User(user_input))
	mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
	mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

	# MLP part 
	mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
	mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
	mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
	for idx in range(1, num_layer):
		layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
		mlp_vector = layer(mlp_vector)

	# Concatenate MF and MLP parts
	#mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
	#mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
	predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
	
	# Final prediction layer
	prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)

	model = Model(input=[user_input, item_input], 
				  output=prediction)
	
	return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
	# MF embeddings
	gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
	gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
	model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
	model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
	
	# MLP embeddings
	mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
	mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
	model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
	model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
	
	# MLP layers
	for i in range(1, num_layers):
		mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
		model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
		
	# Prediction weights
	gmf_prediction = gmf_model.get_layer('prediction').get_weights()
	mlp_prediction = mlp_model.get_layer('prediction').get_weights()
	new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
	new_b = gmf_prediction[1] + mlp_prediction[1]
	model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
	return model

def get_train_instances(train, num_negatives):
	user_input, item_input, labels = [],[],[]
	num_users = train.shape[0]
	for (u, i) in train.keys():
		# positive instance
		user_input.append(u)
		item_input.append(i)
		labels.append(1)
		# negative instances
		for t in range(num_negatives):
			j = np.random.randint(num_items)
			# while train.has_key((u, j)):
			while (u,j) in train:
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	return user_input, item_input, labels

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

	zero_ratings=[]
	positive_ratings=[]

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
				if(float(ratings_of_users[i][key][0])==0):
					zero_ratings.append([user_id, item_id])
				else:
					positive_ratings.append([user_id, item_id])

	return np.asarray(user_item_matrix), np.asarray(ratings), zero_ratings, positive_ratings





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

def get_ratings(RATINGS_MATRIX):
	h = len(RATINGS_MATRIX)
	w = len(RATINGS_MATRIX[0])
	pos_rating_list = [] #user Id, item id  
	neg_rating_list = [] 
	for i in range(h):
		for j in range(w):
			if RATINGS_MATRIX[i][j] > 0:
				pos_rating_list.append([i , j]) 
			else:
				neg_rating_list.append([i , j]) 


	return pos_rating_list , neg_rating_list





def train_test_split(self, test_size):
	train=[]
	test=[]

	test_len=test_size*len(self.ratings)

	for i in range(int(test_len)):
		test.append(self.ratings[i])

	for i in range(int(test_len), len(self.ratings)):
		train.append(self.ratings[i])

	return np.asarray(train), np.asarray(test)

# USERS=retrieve_users()
# MOVIES=retrieve_movies()
# MOVIE_IDS=define_movie_id(MOVIES)
# USER_IDS, USER_NAMES=define_user_id(USERS)
# RATINGS_MATRIX, RATINGS, NEGATIVE_RATINGS, POSITIVE_RATINGS=retrieve_ratings_matrix(MOVIES, USERS, MOVIE_IDS, USER_IDS, USER_NAMES)
# print(RATINGS_MATRIX.shape, RATINGS.shape)

RATINGS_MATRIX=pickle.load(open(RATINGS_MATRIX_FILE, 'rb'))
RATINGS=ratings_for_project(RATINGS_MATRIX)
NEGATIVE_RATINGS, POSITIVE_RATINGS = get_ratings(RATINGS_MATRIX)[::-1]

ITERATIONS=[]
LOSSES=[]
NDCGS=[]


args=parse_args()
path="./"
num_epochs=args.epochs
batch_size=args.batch_size
mf_dim=args.num_factors
layers=eval(args.layers)
reg_mf=args.reg_mf
reg_layers=eval(args.reg_layers)
num_negatives=args.num_neg
learning_rate=args.lr
learner=args.learner
verbose=args.verbose
mf_pretrain=args.mf_pretrain
mlp_pretrain=args.mlp_pretrain
		
topK=10
evaluation_threads=1
print("NeuMF arguments: %s " %(args))
model_out_file = './ _NeuMF_%d_%s_%d.h5' %(mf_dim, args.layers, time())

# Loading data
t1=time()

# pos_ratings,neg_ratings=load_rating_file_as_list('./data/ratings_correctFormat.json')
pos_ratings, neg_ratings=POSITIVE_RATINGS, NEGATIVE_RATINGS
test_size=int(0.2*len(pos_ratings))
testRatings=pos_ratings[:test_size] 
testNegatives=neg_ratings[:test_size] 
train_pos=pos_ratings[test_size:]
train_neg=neg_ratings[test_size:] 
train=sp.dok_matrix((len(RATINGS_MATRIX), len(RATINGS_MATRIX[0])), dtype=np.float64)
for i in train_pos:
	train[i[0], i[1]] = 1.0

num_users,num_items = train.shape
print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
	  %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

# Build model
model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
if learner.lower() == "adagrad": 
	model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
elif learner.lower() == "rmsprop":
	model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
elif learner.lower() == "adam":
	model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
else:
	model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

# Load pretrain model
if mf_pretrain != '' and mlp_pretrain != '':
	gmf_model = GMF.get_model(num_users,num_items,mf_dim)
	gmf_model.load_weights(mf_pretrain)
	mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
	mlp_model.load_weights(mlp_pretrain)
	model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
	print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))
	
# Init performance
(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
best_hr, best_ndcg, best_iter = hr, ndcg, -1
if args.out > 0:
	model.save_weights(model_out_file, overwrite=True) 
	
# Training model
for epoch in range(num_epochs):
	t1 = time()
	# Generate training instances
	user_input, item_input, labels = get_train_instances(train, num_negatives)
	
	# Training
	hist = model.fit([np.array(user_input), np.array(item_input)], #input
					 np.array(labels), # labels 
					 batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
	t2 = time()
	
	# Evaluation
	if epoch %verbose == 0:
		(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
		hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
		ITERATIONS.append(epoch)
		LOSSES.append(loss)
		NDCGS.append(ndcg)
		print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
			  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
		if hr > best_hr:
			best_hr, best_ndcg, best_iter = hr, ndcg, epoch
			if args.out > 0:
				model.save_weights(model_out_file, overwrite=True)

plt.plot(ITERATIONS, LOSSES)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(ITERATIONS, NDCGS)
plt.xlabel('Epochs')
plt.ylabel('NDCG')
plt.show()

print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
if args.out > 0:
	print("The best NeuMF model is saved to %s" %(model_out_file))
