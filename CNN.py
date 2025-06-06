import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load images and label them
base_dir = r'Your Address'
class1_dir = os.path.join(base_dir, 'Class1')
class2_dir = os.path.join(base_dir, 'Class2')

class1_images = [os.path.join(class1_dir, img) for img in os.listdir(class1_dir)]
class2_images = [os.path.join(class2_dir, img) for img in os.listdir(class2_dir)]

# Combine images and labels
images = class1_images + class2_images
labels = ['0'] * len(class1_images) + ['1'] * len(class2_images)

# Create a DataFrame
data = {'image_path': images, 'label': labels}
df = pd.DataFrame(data)

# Split dataset into train, validation, and test sets
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Preprocess images
image_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='image_path', y_col='label', target_size=image_size, batch_size=batch_size, class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='image_path', y_col='label', target_size=image_size, batch_size=batch_size, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='image_path', y_col='label', target_size=image_size, batch_size=batch_size, class_mode='binary')

# Define CNN architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("Test Accuracy:", test_acc)

# Save the trained model
model.save("my_cnn_model.h5")
print("Model saved successfully as 'my_cnn_model.h5'")
