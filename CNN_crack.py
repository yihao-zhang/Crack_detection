import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPool2D, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from keras.preprocessing.image import ImageDataGenerator

######gpu option
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

NAME = 'crack-cnn-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
train_dir = 'input_data'
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (227,227),
        batch_size = 100,   ##20
        class_mode = 'categorical'
        )


model = Sequential()

model.add( Conv2D(24, input_shape = (227,227,3),
                  kernel_size=(20,20), strides=(2,2),
                  activation='relu', kernel_regularizer = l2(0.01)) )
 
model.add( BatchNormalization())
model.add(MaxPool2D((7,7), strides=(2,2)))

model.add( Conv2D(48, kernel_size=(15,15), strides=(2,2),
                  activation='relu',kernel_regularizer = l2(0.01)) )

model.add( BatchNormalization())
model.add(MaxPool2D((4,4), strides=(2,2)))

model.add( Conv2D(96, kernel_size=(8,8), strides=(2,2), 
                  activation='relu', kernel_regularizer = l2(0.01)) )

model.add( BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(2, activation = 'softmax'))

model.compile(
        optimizer=SGD(momentum=1e-4, decay=0.9),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'] 
        )

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 160,
        epochs =10,
        validation_data = train_generator,
        validation_steps = 20,
        callbacks=[tensorboard]
        )

#
model.save('crack_model.h5')


