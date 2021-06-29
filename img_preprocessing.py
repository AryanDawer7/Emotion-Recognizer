from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

############################ TRAIN
data = pd.read_csv("datasets/train.csv")

width, height = 48, 48

datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

y = np.array(data['emotion'])

np.save('train_X.npy',X)
np.save('train_y.npy',y)

print(f'train_X.npy shape = {X.shape}')
print(f'train_y.npy shape = {y.shape}')

############################ TEST
data2 = pd.read_csv("datasets/test.csv")

datapoints = data2['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

np.save('test_X.npy',X)

print(f'test_X.npy shape = {X.shape}')

"""
Visual Representation

classes = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

img = plt.imshow(X[56])
plt.title(classes[y[56]])
plt.show()
"""