# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:19:57 2025

@author: Ajay
"""

import os
import shutil
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# STEP 2: Data Paths
base_dir = '/content/Datasplitted_unzip'
train_dir = os.path.join(base_dir, '/content/Datasplitted_unzip/Datasplitted/train')
val_dir = os.path.join(base_dir, '/content/Datasplitted_unzip/Datasplitted/val')
test_dir = os.path.join(base_dir, '/content/Datasplitted_unzip/Datasplitted/test')

# STEP 3: Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# STEP 4: Build Model with DenseNet121
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# STEP 5: Train Model
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('embryo_model.h5', save_best_only=True)
earlystop_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# STEP 6: Evaluate on Test Set
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy:.2f}')

# STEP 7: Save Model
model.save('embryo_model_final.h5')

# STEP 8: Flask API for Inference
app = Flask(__name__)
model = load_model('embryo_model_final.h5')
class_indices = train_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img = load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = inv_class_indices[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
