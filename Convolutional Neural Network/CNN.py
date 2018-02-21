from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#step1 : Convolution




classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'  ))   #Using 32 feature detectprs /filters with dimension 3* 3

#step 2 : Max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3: Flatten
classifier.add(Flatten())


# step4:Full Connection
classifier.add((Dense(output_dim = 128 , activation = 'relu')))
classifier.add((Dense(output_dim = 1 , activation = 'sigmoid')))


#compiling the cnn
classifier.compile(optimizer='adam' , loss= 'binary_crossentropy', metrics = ['accuracy'])




#Part 2: Fitting the CNN to an image
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
            training_set,
            steps_per_epoch=8000, #number of images in our training set
            epochs=25,
            validation_data=test_set,
            validation_steps=2000) #number of images in the test set




