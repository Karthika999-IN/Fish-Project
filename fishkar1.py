import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# ✅ Define Paths
train_dir = r"C:\Users\karthika\Desktop\Fish project_442025\data\train"
test_dir = r"C:\Users\karthika\Desktop\Fish project_442025\data\test"
best_model_path = r"C:\Users\karthika\Desktop\Fish project_442025\best_model.h5"

# ✅ Image Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ✅ Fetch Class Names
class_names = list(train_generator.class_indices.keys())

# ✅ Function to Build & Train Models
def build_and_train_model(base_model, model_name):
    base_model.trainable = False  # Freeze base model
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_generator, validation_data=test_generator, epochs=5)

    model.save(f"C:/Users/karthika/Desktop/Fish project_442025/{model_name}.h5")

    return model, history

# ✅ Train & Evaluate Multiple Models
models = {
    "VGG16": VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "ResNet50": ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "MobileNet": MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "InceptionV3": InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "EfficientNetB0": EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
}

histories = {}
model_accuracies = {}

for model_name, base_model in models.items():
    print(f"\n🔵 Training {model_name}...\n")
    trained_model, history = build_and_train_model(base_model, model_name)
    histories[model_name] = history
    model_accuracies[model_name] = max(history.history["accuracy"])

# ✅ Select the Best Model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = load_model(f"C:/Users/karthika/Desktop/Fish project_442025/{best_model_name}.h5")
best_model.save(best_model_path)
print(f"\n✅ Best Model: {best_model_name} (Saved as {best_model_path})\n")

# ✅ Evaluate Model Performance
y_true = test_generator.classes
y_pred_probs = best_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# ✅ Classification Report
print("\n🔹 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ✅ Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ Deploy Model with Streamlit
def predict_fish(image_path, model):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# ✅ Streamlit Web App
st.title("🐟 Fish Classification Model")
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    predicted_class, confidence = predict_fish("temp.jpg", best_model)

    st.write(f"**Prediction:** {predicted_class} 🎯")
    st.write(f"**Confidence:** {confidence:.2f}")

