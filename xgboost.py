import numpy as np
import xgboost as xgb

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

# XGBoost Classification
model = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=500, reg_alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
# Correct kaggle output format
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