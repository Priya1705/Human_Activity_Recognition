import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


np.random.seed(0)

data=pd.read_csv('train.csv')
feature_set=data.iloc[:,:-2]
labels=data.iloc[:,-2]

feature_set=pd.DataFrame(feature_set)
labels=pd.DataFrame(labels)

feature_set.to_csv("feature_set.csv" ,header=False ,index=False)
labels.to_csv("labels.csv" , header=False ,index=False)

one_hot_labels = np.zeros((7351, 31))

feature_set=pd.read_csv('feature_set.csv')
labels=pd.read_csv('labels.csv')

feature_set=feature_set.values.reshape(7351,561)
labels = labels.values.reshape(7351, 1)

# print(labels[30])

# print(one_hot_labels[0,:])

for i in range(7351):
    one_hot_labels[i, labels[i]] = 1


one_hot_labels=pd.DataFrame(one_hot_labels)
one_hot_labels.to_csv("onehotlabels.csv" , header=False ,index=False)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 4
output_labels = 31

wh = np.random.rand(attributes,hidden_nodes)
bh = np.random.randn(hidden_nodes)
wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 10e-4
y=[]

# print(np.shape(wh))
# print(np.shape(bh))
# print(np.shape(wo))
# print(np.shape(bo))
# print(np.shape(feature_set))

error_cost = []

for epoch in range(500):
############# feedforward

    # Phase 1
    print(epoch)
    zh = np.dot(feature_set, wh) #+ bh
    ah = sigmoid(zh)

    # Phase 2
    zo = np.dot(ah, wo) #+ bo
    ao = softmax(zo)

    if epoch==499:
        y.append(ao)
        # print(np.shape(y))
# ########## Back Propagation

# # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - one_hot_labels), 2)))
    # print(error_out.sum())

    dcost_dao = ao - one_hot_labels
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh) 
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)



# ########## Phase 1


#     dcost_dzo = ao - one_hot_labels
#     dzo_dwo = ah

#     dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

#     dcost_bo = dcost_dzo

# ########## Phases 2

#     dzo_dah = wo
#     dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
#     dah_dzh = sigmoid_der(zh)
#     dzh_dwh = feature_set
#     dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

#     dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    # bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    # bo -= lr * dcost_bo.sum(axis=0)

    # if epoch % 200 == 0:
    #     loss = np.sum(-one_hot_labels * np.log(ao))
    #     print('Loss function value: ', loss)
    #     error_cost.append(loss)





print("hello")
# print(y)
# print(np.shape(y))

arr=[]

print(y)

for temp in y:
    for elem in temp:
        arr.append((elem))

print(np.shape(arr))
print(arr)
df = pd.DataFrame(arr)
df.to_csv("labelpredict.csv",index=False, header=False)