import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import tensorflow_hub as hub
import time
import argparse
from PIL import Image

batch_size = 32
image_size = 224
class_names = {}

def process_image(image): 
   
    image = tf. convert_to_tensor(image)
    image = tf.image.resize(image, [224,224], preserve_aspect_ratio=False)
    image /= 255
    image=image.numpy()
    
    return image
       

    
def predict(image_path,model,top_k):
    
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    
    probs = model.predict(processed_test_image)
    
    top_k_values, top_k_indices = tf.math.top_k(probs, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    
    
    return top_k_values[0], top_k_indices[0]



if __name__ == '__main__':
    
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k',type=int)
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    
      
    image_path = args.arg1    
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k
    
    if top_k == None:
        top_k=5
     
    probabilities, classes = predict(image_path, model, top_k)
    
    if args.category_names == None:       
        print(probabilities)
        print(classes)
    
    else:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    
        cn=[]

        for i in classes:
            cn.append(class_names[str(i+1)])
            
        print(probabilities)
        print(cn)




