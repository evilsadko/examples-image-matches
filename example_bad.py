# -*- coding: utf-8 -*-
import sys, cv2, os
import numpy as np
import keras
from keras.preprocessing import image as image_utils

def imgs_v(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

class DATA(object):
   def __init__(self):
       self.file = []

   def parseIMG(self, dir_name):
       path = f"{dir_name}/"
       print ("PARSING",path)
       for r, d, f in os.walk(path):
           for ix, file in enumerate(f): 
                      if ".png" in file: 
                          self.file.append(os.path.join(r, file))
                      if ".jpg" in file: 
                          self.file.append(os.path.join(r, file))

def deep_vector(x):
       t_arr = image_utils.load_img(x, target_size=(224, 224)) 
       t_arr = image_utils.img_to_array(t_arr)
       t_arr = np.expand_dims(t_arr, axis=0)
       processed_img = preprocess(t_arr)
       preds = model.predict(processed_img)
       return preds

def similarity(vector1, vector2):
        return np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True), np.linalg.norm(vector2.T, axis=0, keepdims=True))

def func_rec(ID):
    if len(arr.file) != 0:
        for i in range(len(arr.file)):
            if ID not in G.keys():
                    G[ID] = [arr.file[i]]
                    preds0 = deep_vector(arr.file[i])
                    del arr.file[i]
            else:
                try:
                    preds1 = deep_vector(arr.file[i])
                    KEF = similarity(preds0, preds1)
                    if KEF[0]>tresh:
                         G[ID].append(arr.file[i])
                         del arr.file[i]
                except IndexError: 
                    if len(arr.file) == 0:
                         break
                    print (i, len(arr.file), len(G.keys()))
                    ID += 1
                    func_rec(ID)

if __name__ == '__main__':
     if len(sys.argv)>1:
        arr = DATA()
        arr.parseIMG(sys.argv[1])

        model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max')
        preprocess = keras.applications.vgg16.preprocess_input
        tresh = .68

        G = {}
        preds0 = 0
        func_rec(0)
        if len(list(G.keys())) != 0:
            os.mkdir(f"out") 
        for U in list(G.keys()):
            os.mkdir(f"out/{U}")
            for UU in G[U]:
               im = cv2.imread(UU)
               name = UU.split("/")[-1]
               cv2.imwrite(f"out/{U}/{name}", im)


