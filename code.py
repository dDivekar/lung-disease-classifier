import numpy as np
import pandas as pd


from google.colab import files,drive
import zipfile

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

uploaded=files.upload()

zip_path='archive.zip'
with zipfile.ZipFile(zip_path,'r') as zip_ref:
  zip_ref.extractall('dataset')

def load_images_from_folder(folder,label,img_size=(150,150)):
  data=[]
  for filename in os.listdir(folder):
    img_path=os.path.join(folder,filename)
    img=cv2.imread(img_path)
    if img is not None :
      img=cv2.resize(img,img_size)
      data.append([img,label])
  return data

print("content of dataset directory : ")
print(os.listdir('dataset'))
Fibrosis_folder="dataset/Fibrosis"
Normal_folder="dataset/Normal"
Pneumonia_folder="dataset/Pneumonia"

print(f"using Fibrosis folder : {Fibrosis_folder}")
print(f"using Normal folder : {Normal_folder}")
print(f"using Pneumonia folder : {Pneumonia_folder}")

fibrosis_data=load_images_from_folder(Fibrosis_folder,0)
normal_data=load_images_from_folder(Normal_folder,1)
pneumonia_data=load_images_from_folder(Pneumonia_folder,2)

data=fibrosis_data+normal_data+pneumonia_data
np.random.shuffle(data)
X = np.array([item[0] for item in data],
				dtype="float32") / 255.0
							# normalize
y = np.array([item[1] for item in data])

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set=train_datagen.flow_from_directory(
    'dataset',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. CNN Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model # Import Model class for functional API
from tensorflow.keras.models import Sequential # Keep Sequential for its definition

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)   # Changed to 3 output neurons with softmax for 3 classes

model = Model(inputs=base_model.input, outputs=output)

# 5. Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import tensorflow as tf

# Resize X_train and X_test to match the model's expected input shape (224, 224, 3)
X_train_resized = tf.image.resize(X_train, (224, 224)).numpy()
X_test_resized = tf.image.resize(X_test, (224, 224)).numpy()

# 6. Train
model.fit(X_train_resized, y_train, epochs=10,batch_size=32, validation_data=(X_test_resized, y_test))
# 7. Save the Model
model.save("lung_disease_manual_cnn.h5")

import random
import matplotlib.pyplot as plt
import numpy as np

# 1. Class labels (match your dataset)
class_names = ["Fibrosis", "Normal", "Pneumonia"]

# 2. Pick 10 random indices
indices = random.sample(range(len(X_test)), 10)

plt.figure(figsize=(15, 5))

for i, idx in enumerate(indices):
    img = X_test[idx]
    true_label = y_test[idx]

    # Add batch dimension
    pred_probs = model.predict(img.reshape(1, 150, 150, 3), verbose=0)[0]

    # Get predicted class (highest probability)
    pred_label = np.argmax(pred_probs)

    # Plot image
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')

    plt.title(
        f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}",
        fontsize=9
    )

plt.tight_layout()
plt.show()
