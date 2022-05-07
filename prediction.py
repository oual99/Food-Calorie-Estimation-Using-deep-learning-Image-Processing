from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import numpy as np


img_size = 150
#img_path = "/Users/mac/Documents/SMI/SMIS6/PFE/imageSeg/img/2.jpg"

def get_food_name(argument):
    tab = {
        1: "Apple",
        2: "Banana",
        3: "Cucumber",
        4: "Onion",
        5: "Orange",
        6: "Tomato"
    }
    return (tab[argument])


def getPrediction(model, img_path):
    img = image.load_img(img_path, target_size = (img_size,img_size))
    #plt.imshow(img)
    #plt.show()
    X = image.img_to_array(img)
    X = X.reshape((1, )+X.shape)
    val = model.predict(X)
    result = np.argmax(val)
    label = get_food_name(result+1)
    return label,round(val[0][result]*100,2)
