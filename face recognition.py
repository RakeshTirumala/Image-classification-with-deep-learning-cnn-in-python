import os
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image

people = []
for i in os.listdir(r'/content/drive/MyDrive/Projects/FR-1/Dataset/train'):
  people.append(i)
print(people)

TrainingImagePath = (r'/content/drive/MyDrive/Projects/FR-1/Dataset/train')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(TrainingImagePath, 
                                                 target_size=(64,64), 
                                                 batch_size=32, 
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(TrainingImagePath,
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode = 'categorical')
test_set.class_indices

TrainClasses = training_set.class_indices

ResultMap = {}

for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
  ResultMap[faceValue]=faceName
  
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

print("Mapping of Face and its ID",ResultMap)

OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

classifier = Sequential()

classifier.add(Convolution2D(32, kernel_size=(5, 5), 
                             strides=(1,1), 
                             input_shape=(64,64,3), 
                             activation='relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(64, kernel_size=(5, 5), 
                             strides=(1, 1), 
                             activation='relu'))

classifier.add(Flatten())

classifier.add(Dense(64, activation='relu'))

classifier.add(Dense(OutputNeurons, activation='softmax'))


classifier.compile(loss='categorical_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])

history = classifier.fit(training_set,
                         batch_size=25, 
                         steps_per_epoch=4,
                         epochs=4,
                         validation_data=test_set)

Imagepath = (r'/content/drive/MyDrive/Projects/FR-1/Dataset/test/Rakesh/IMG_20210617_171534.jpg')

test_image = image.load_img(Imagepath, target_size=(64,64))

test_image = image.img_to_array(test_image)
 
test_image = np.expand_dims(test_image,axis=0)
 
result=classifier.predict(test_image,verbose=0)

print('Prediction is: ',ResultMap[np.argmax(result)])
