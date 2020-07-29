import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv('train.csv') 
# dataset.head(10) #Return 10 rows of data

X=dataset.iloc[:,:-2]
y=dataset.iloc[:,-2]

X=pd.DataFrame(X)
y=pd.DataFrame(y)

X.to_csv("feature_set.csv" ,header=False ,index=False)
y.to_csv("labels.csv" , header=False ,index=False)

#Normalizing the data
sc = StandardScaler()
X = sc.fit_transform(X)

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)



# Neural network
model = Sequential()
model.add(Dense(4, input_dim=561, activation='relu'))
# model.add(Dense(12, activation=’relu’))
model.add(Dense(21, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=1000, batch_size=50)


y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)


# history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)



