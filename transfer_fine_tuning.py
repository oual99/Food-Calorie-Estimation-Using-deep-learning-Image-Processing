from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras import models, layers

import numpy as np
import matplotlib.pyplot as plt

datasetPath = "/Users/mac/Documents/SMI/SMIS6/PFE1/TL_using_ResNet/all_dataset/train/"
val11 = "/Users/mac/Documents/SMI/SMIS6/PFE1/TL_using_ResNet/all_dataset/val1/"
test = "/Users/mac/Documents/SMI/SMIS6/test/"
img_size = 100
batch_size = 32

train = ImageDataGenerator(preprocessing_function = preprocess_input,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip = True,
                            height_shift_range=0.5,
                            rotation_range=0.5,
                          )
validation = ImageDataGenerator(preprocessing_function = preprocess_input)


train_dataset = train.flow_from_directory(datasetPath,
                                          target_size = (img_size,img_size),
                                          batch_size = batch_size,
                                          class_mode = 'categorical',
                                          color_mode = 'rgb',
                                         )
validation_dataset = validation.flow_from_directory(val11,
                                                target_size = (img_size,img_size),
                                                batch_size = batch_size,
                                                class_mode = 'categorical',
                                                color_mode = 'rgb',
                                                )

early_stop = EarlyStopping(monitor = 'val_loss',
                                     patience = 10,
                                     verbose = 1,
                                    )

#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

callbacks_list = [early_stop, log_csv]


model = models.Sequential([ResNet50(include_top=False, weights = 'imagenet', pooling = 'avg'),
                           
                           layers.Dense(512,activation = 'relu'),
                           layers.Dropout(0.2),
                           layers.Dense(6,activation = 'softmax')  
                                                     ])

model.layers[0].trainable = False

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])


history = model.fit(train_dataset,
          steps_per_epoch = 279,
          epochs = 25,
          validation_data = validation_dataset,
          validation_steps = 39,
          callbacks = callbacks_list
         )

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

model.save('saved_model_transfer_learning/')

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

for i in os.listdir(test_path):
    img = image.load_img(test_path+ i,target_size = (img_size,img_size))
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

early_stop = EarlyStopping(monitor = 'val_loss',
                                     patience = 2,
                                     verbose = 1,
                                    )


#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

callbacks_list1 = [early_stop, log_csv]

model.layers[0].trainable = True


model.compile(optimizer= Adam(1e-5),  # Very low learning rate
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset,
          steps_per_epoch = 279,
          epochs = 6,
          validation_data = validation_dataset,
          validation_steps = 39,
          callbacks = callbacks_list1
         )

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


model.save('saved_model_fine_tuning/')


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



