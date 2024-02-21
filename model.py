import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Mount Google Drive (if your dataset is stored there)
from google.colab import drive
drive.mount('/content/drive')

# Define paths to your dataset
train_data_dir = '/content/drive/MyDrive/alzheimerdata/train'
validation_data_dir = '/content/drive/MyDrive/alzheimerdata/test'

# Set up data generators for training and validation
batch_size = 32
img_size = (176, 208)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dåir,
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=img_size,
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))  # Assuming 4 classes: non-dementia, minor, mild, major

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the best model during training
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
epochs = 20
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    callbacks=[checkpoint])

# Save the final model
model.save('final_model.h5')
loaded_model = tf.keras.models.load_model('final_model.h5')  # Use the path where your final model is saved

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array
def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_labels = ['non-dementia', 'minor dementia', 'mild dementia', 'major dementia']
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class, prediction[0]
from google.colab import files

uploaded = files.upload()

img_path = list(uploaded.keys())[0]
predicted_class, confidence = predict_image(loaded_model, img_path)


img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Prediction: {predicted_class} (Confidence: {confidence.max():.2f})')
plt.show()