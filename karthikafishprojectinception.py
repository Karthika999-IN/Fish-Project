# 🚀 Import section
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# ✅ Define Paths
train_dir = r"C:\Users\karthika\Desktop\Fish project_442025\data\train"
test_dir = r"C:\Users\karthika\Desktop\Fish project_442025\data\test"
model_save_path = r"C:\Users\karthika\Desktop\Fish project_442025\best_fish_model_inceptionv3.h5"

# ✅ Image Parameters
IMG_SIZE = (224, 224)  # InceptionV3 prefers (299,299) but we keep (224,224) for uniformity
BATCH_SIZE = 32

# ✅ Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ Fetch Class Names
class_names = list(train_generator.class_indices.keys())

# ✅ Build Model with Fine-Tuning
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

# Freeze first few layers (optional)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Build new model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,
    verbose=1
)

# ✅ Save the model
model.save(model_save_path)
print(f"✅ Model saved at {model_save_path}")

# ✅ Evaluate the model
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n🔹 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ✅ Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ Streamlit App
st.title("🐟 Fish Species Classifier - InceptionV3")

uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    # Predict
    img = load_img("temp.jpg", target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"🎯 Predicted Class: **{predicted_class}**")
    st.info(f"🔵 Confidence Score: **{confidence:.2f}**")
