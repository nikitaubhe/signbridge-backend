import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import model_from_json

print("Loading model...")
with open("model.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("model.h5")
print("Model loaded.")

seq = np.zeros((1, 30, 63))
print("Predicting...")
try:
    res = model(seq, training=False)
    print("Done! Result shape:", res.shape)
except Exception as e:
    print("Error:", e)
