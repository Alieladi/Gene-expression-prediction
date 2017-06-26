import numpy as np
from keras.models import load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential


X_train = np.loadtxt("competition/x_train.csv",delimiter=',',skiprows=1)
y_train = np.loadtxt("competition/y_train.csv",delimiter=',',skiprows=1)
X_test = np.loadtxt("competition/x_test.csv",delimiter=',',skiprows=1)

X_train =  X_train[:,1:]  #Training data
X_test =  X_test[:,1:]  #Training data
y_train = y_train[:,1:].ravel() #Labels
# Rmk : There are 100 variables for each case
gene_train = X_train.shape[0]/100
gene_test = X_test.shape[0]/100

print(gene_train) 
X_train = np.split(X_train,gene_train) # Divide X_train into gene_train equal arrays
X_test = np.split(X_test,gene_test) # Divide X_test into gene_test equal arrays

X_train = np.array([x.ravel() for x in X_train])
X_test = np.array([x.ravel() for x in X_test])
y_train = np.array(y_train)

#Correct format for the neural network below
y_train = np.vstack([1-y_train, y_train]).T


def neuralNet():
	N = 5 # Number of feature maps
	w, h = 1, 5 # Conv. window size
	model = Sequential()
	model.add(Convolution2D(nb_filter = N,
	nb_col = w,
	nb_row = h,
	border_mode = 'same',
	activation = 'relu',
	input_shape = (100,1,5)))
	model.add(MaxPooling2D((2,1)))
	
	model.add(Convolution2D(nb_filter = N,
	nb_col = w,
	nb_row = h,
	border_mode = 'same',
	activation = 'relu'))
	model.add(MaxPooling2D((2,1)))
	model.add(Flatten())
	model.add(Dense(2, activation = 'sigmoid'))

	model.compile(loss='mean_squared_error', optimizer='sgd')
	model.fit(X_train, y_train, nb_epoch=20, batch_size=16) # takes a few seconds
	model.save("neuro_model.h5")
	
	y_pred = model.predict_proba(X_test)
	y_pred_classes = model.predict_classes(X_test)
	y_pred_kaggle = np.array(list(map(lambda x: x[1],y_pred)))
	
	
	geneId=0
	f = open("output.csv","w")
	f.write("GeneId,prediction")
	f.write("\n")
	for i in y_pred_kaggle:
		geneId = geneId + 1
		f.write(str(geneId)+","+str(i))
		f.write("\n")

	f.close()
	
neuralNet()

