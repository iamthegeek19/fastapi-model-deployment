from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib.pyplot import imread

app = FastAPI()

class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage']

def load_and_prep_image(image):
  img = tf.image.resize(image, [224,224])
  return img

data = [
   {
      'name': 'clean',
      'img_path': './images/bird.jpg'
   },
   {
      'name': 'bird',
      'img_path': './images/clean.jpg'
   },
   {
      'name': 'dust',
      'img_path': './images/dust.jpg'
   },
   {
      'name': 'electrical',
      'img_path': './images/electrical.jpg'
   },
   {
      'name': 'physical',
      'img_path': './images/physical.jpg'
   }
]

results = []

@app.get('/')
def test():
   return 'chitima chafergurson'

@app.get('/predict')
def hello():
    model = load_model('native_model.h5')
    for i in range(len(data)):
        img = imread(data[i]['img_path'])
        pred_img = load_and_prep_image(img)
        pred_prob = model.predict(tf.expand_dims(pred_img, axis=0))
        res = tf.argmax(pred_prob, axis=1).numpy()[0]
        results.append({
           data[i]['name']: class_names[res]
        })
    
    print(results)
    return {'data': results}
    