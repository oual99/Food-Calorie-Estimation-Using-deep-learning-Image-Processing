import numpy as np         # dealing with arrays
import os                  # dealing with directories
import cv2
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import models, layers

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_size = 100
batch_size = 32

training_path = "/Users/mac/Documents/SMI/SMIS6/PFE1/TL_using_ResNet/all_dataset/train/"
validation_path = "/Users/mac/Documents/SMI/SMIS6/PFE1/TL_using_ResNet/all_dataset/val1/"

train = ImageDataGenerator(rescale=1./255,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           vertical_flip = True,
                           height_shift_range=0.5,
                          )


train_dataset = train.flow_from_directory(datasetPath,
                                          target_size = (img_size,img_size),
                                          batch_size = batch_size,
                                          class_mode = 'categorical',
                                          color_mode = 'rgb',
                                         )
validation_dataset = train.flow_from_directory(val11,
                                          target_size = (img_size,img_size),
                                          batch_size = batch_size,
                                          class_mode = 'categorical',
                                          color_mode = 'rgb'
                                          )

train_dataset.class_indices

early_stop = EarlyStopping(monitor = 'val_loss',
                                     patience = 6,
                                     verbose = 1,
                                    )

#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

callbacks_list = [early_stop, log_csv]

model = models.Sequential([layers.Conv2D(16,(3,3),activation = 'relu', input_shape = (img_size,img_size,3)),
                           layers.MaxPool2D(2,2),
                           layers.Conv2D(32,(3,3), activation = 'relu'),
                           layers.MaxPool2D(2,2),
                           layers.Dropout(0.5),
                           layers.Conv2D(64,(3,3), activation = 'relu'),
                           layers.MaxPool2D(2,2),
                           layers.Conv2D(128,(3,3), activation = 'relu'),
                           layers.Dropout(0.5),
                           layers.MaxPool2D(2,2),
                           layers.Flatten(),
                           layers.Dense(256,activation = 'relu'),
                           layers.Dense(6,activation = 'softmax')  
                          ])

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

history = model.fit(train_dataset,
          steps_per_epoch = 279,
          epochs = 15,
          validation_data = validation_dataset,
          validation_steps = 39,
          callbacks = callbacks_list
         )

model.save("saved_model/")

test_path = "/Users/mac/Documents/SMI/SMIS6/test/"

for i in os.listdir(test):
    img = image.load_img(test+ i,target_size = (img_size,img_size))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = X.reshape((1, )+X.shape)
    val = model.predict(X)
    
    print(val)
    
    res = np.argmax(val)
    print(round(val[0][res]*100,2),'%')
    index = res+1
    get_food_name(index)
    print(index)
    print(i)


def get_food_name(argument):
    tab = {
        1: "Apple",
        2: "Banana",
        3: "Cucumber",
        4: "Onion",
        5: "Orange",
        6: "Tomato"
    }
    print (tab[argument])

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()






