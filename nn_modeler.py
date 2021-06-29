import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split

train_X = np.load('train_X.npy')
train_y = np.load('train_y.npy')

train_X, test_X, train_y, test_y = train_test_split(train_X,train_y,test_size=0.1)

width, height = 48, 48
num_features = 64
num_labels = 7

epochs = 30
batch_size = 64
learning_rate = 0.001

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))
model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))    

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))    

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))    

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))    

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

adam = keras.optimizers.Adam(lr = learning_rate)

model.compile(optimizer=adam, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])    

print(model.summary())

lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)

early_stopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=6, mode='auto')

checkpointer = keras.callbacks.ModelCheckpoint('Model/weights.hd5', monitor='val_loss', verbose=1, save_best_only=True)


model.fit(train_X,train_y,batch_size=batch_size,shuffle=True,validation_split = 0.2,epochs=epochs,callbacks=[lr_reducer, checkpointer, early_stopper])

results = model.evaluate(test_X, test_y)
print(results)

model.save("model_emotion_recog.h5")