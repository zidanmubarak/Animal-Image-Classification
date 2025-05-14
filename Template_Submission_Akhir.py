# %% [markdown]
# # Proyek Klasifikasi Gambar: Animals-10
# - **Nama:** Zidan Mubarak
# - **Email:** zidanmubarak00@gmail.com
# - **ID Dicoding:** zidanmubarak

# %% [markdown]
# ## Import Semua Packages/Library yang Digunakan

# %%
# !pip install tensorflowjs

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shutil
import glob
import random
import tensorflowjs as tfjs
from PIL import Image
from tqdm.notebook import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
# mengatur seed untuk reproduksibilitas
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Data Loading

# %%
# mengupload dataset dari local
from google.colab import drive
drive.mount('/content/drive')

# path ke dataset animal-10
dataset_path = '/content/drive/MyDrive/dicoding/pengembangan ML/submission 2/Animals-10'

# kelas hewan yang dipilih
selected_classes = ['chicken', 'dog', 'horse']

# %%
# verifikasi bahwa kelas yang dipilih ada di dataset
available_classes = os.listdir(dataset_path)
print("kelas yang tersedia di dataset:")
print(available_classes)
print("kelas yang dipilih untuk klasifikasi:")
print(selected_classes)

# %%
# membuat dataframe untuk data dan label
def create_dataframe_from_selected_classes(directory, classes):
  data = []
  for category in classes:
    category_dir = os.path.join(directory, category)
    if os.path.isdir(category_dir):
      for img_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
          data.append({
              'image_path': img_path,
              'label': category
          })

  return pd.DataFrame(data)

# %%
# membuat dataframe dari kelas yang dipilih
animals_df = create_dataframe_from_selected_classes(dataset_path, selected_classes)
print(f"\ntotal jumlah data: {len(animals_df)}")
print(animals_df['label'].value_counts())

# %%
# visualisasi sampel gambar dari setiap kelas yang dipilih
plt.figure(figsize=(15, 12))
for i, cls in enumerate(selected_classes):
  # ambil sampel gambar dari kelas
  class_samples = animals_df[animals_df['label'] == cls]['image_path'].sample(min(4, sum(animals_df['label'] == cls))).tolist()

  for j, img_path in enumerate(class_samples):
    img = plt.imread(img_path)
    plt.subplot(len(selected_classes), 4, i*4 + j +1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# visualisasi distribusi kelas
plt.figure(figsize=(10, 6))
sns.countplot(y='label', data=animals_df, order=animals_df['label'].value_counts().index)
plt.title('Distribusi Kelas')
plt.xlabel('Jumlah Gambar')
plt.ylabel('Kelas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Verifikasi bahwa gambar memiliki resolusi yang tidak seragam (tanpa preprocessing)
print("\nMemeriksa resolusi gambar untuk membuktikan dataset memiliki resolusi tidak seragam:")
resolutions = []
sample_images = animals_df.groupby('label').apply(lambda x: x.sample(1)).reset_index(drop=True)

for idx, row in sample_images.iterrows():
  img = Image.open(row['image_path'])
  width, height = img.size
  resolutions.append((width, height))
  print(f"kelas {row['label']}, gambar: {os.path.basename(row['image_path'])}, resolusi: {width}x{height}")

# %%
# Cek apakah semua resolusi sama
is_uniform = len(set(resolutions)) == 1
print(f"\nApakah resolusi gambar seragam? {'Ya' if is_uniform else 'Tidak'}")
if not is_uniform:
    print("Dataset memiliki resolusi gambar yang tidak seragam, sesuai kriteria.")

# %%
# menghitung total gambar
total_images = len(animals_df)
print(f"\ntotal gambar dalam dataset yang dipilih: {total_images}")

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Split Dataset

# %%
# split data memjadi train, test, validation dengan stratifikasi
train_df, temp_df = train_test_split(animals_df, test_size=0.2, stratify=animals_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"train set: {len(train_df)} gambar")
print(f"test set: {len(test_df)} gambar")
print(f"validation set: {len(val_df)} gambar")

# %%
# memeriksa distribusi kelas di setiap set
print("\nDistribusi kelas pada train set:")
print(train_df['label'].value_counts())
print("\nDistribusi kelas pada validation set:")
print(val_df['label'].value_counts())
print("\nDistribusi kelas pada test set:")
print(test_df['label'].value_counts())

# %%
# mendefenisikan parameter
IMG_SIZE = 224 # ukuran gambar sesuai dengan model standar
BATCH_SIZE = 32
NUM_CLASSES = len(selected_classes)

# buat direktori untuk train, test, dan validation
output_base_dir = '/content/drive/MyDrive/dicoding/pengembangan ML/submission 2/animals_dataset_split'
train_output_dir = os.path.join(output_base_dir, 'train')
test_output_dir = os.path.join(output_base_dir, 'test')
val_output_dir = os.path.join(output_base_dir, 'validation')

# hapus direktori jika sudah ada untuk memastikan tidak ada data lama, kalau belum ada direkroti maka dibuat baru
if os.path.exists(output_base_dir):
  shutil.rmtree(output_base_dir)
# else:
#   os.makedirs(output_base_dir)

# buat direktori baru
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# %%
# fungsi untuk menyalin gambar berdasarkan dataframe
def copy_images_simple(df, output_dir):
    # Buat semua folder kelas
    unique_labels = df['label'].unique()
    for label in unique_labels:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Menyalin ke {os.path.basename(output_dir)}"):
        src_path = row.image_path
        dst_path = os.path.join(output_dir, row.label, os.path.basename(src_path))

        # salin gambar ke direktori baru
        shutil.copyfile(src_path, dst_path)

# salin gambar ke direktori yang sesuai
copy_images_simple(train_df, train_output_dir)
copy_images_simple(test_df, test_output_dir)
copy_images_simple(val_df, val_output_dir)

# %%
# verifikasi bahwa gambar telah disalin dengan benar
print("\nJumlah gambar di direktori train:", sum([len(os.listdir(os.path.join(train_output_dir, cls))) for cls in os.listdir(train_output_dir)]))
print("Jumlah gambar di direktori validation:", sum([len(os.listdir(os.path.join(val_output_dir, cls))) for cls in os.listdir(val_output_dir)]))
print("Jumlah gambar di direktori test:", sum([len(os.listdir(os.path.join(test_output_dir, cls))) for cls in os.listdir(test_output_dir)]))

# %% [markdown]
# #### Image Data Generator

# %%
# menggunakan imagedatagenerator untuk augmentasi data dan preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2] # variasi kecerahan
)

# minimal preprocessing untuk test dan validasi
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# %%
# flow generator untuk training
train_generator = train_datagen.flow_from_directory(
    train_output_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_output_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

validation_generator = validation_datagen.flow_from_directory(
    val_output_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# %%
# mendapatkan clss indices
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print("class indices:", class_indices)

# %% [markdown]
# ## Modelling

# %%
# Custom callback untuk menghentikan training jika akurasi melebihi threshold
class AccuracyThresholdCallback(Callback):
    def __init__(self, threshold=0.96):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        if accuracy is not None and accuracy >= self.threshold:
            print(f"\nAkurasi mencapai {accuracy:.4f}, melebihi threshold {self.threshold}. Menghentikan pelatihan.")
            self.model.stop_training = True

# %%
# Model untuk klasifikasi hewan
model = Sequential([
    # layer pertama
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # layer kedua
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # layer ketiga
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # layer keempat
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # layer kelima
    Flatten(),

    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') # 11 kelas
])

# compile model
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# ringkasan model
model.summary()

# %%
# implementasi callback
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss'),
    ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_accuracy'),
    AccuracyThresholdCallback(threshold=0.96)
]

# %%
# jumlah epoch
EPOCHS = 60

# melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=EPOCHS,
    callbacks=callbacks
)

# %% [markdown]
# ## Evaluasi dan Visualisasi

# %%
# Mengevaluasi model pada training set
train_loss, train_acc = model.evaluate(train_generator)
print(f"Training accuracy: {train_acc:.4f}")
print(f"Training loss: {train_loss:.4f}")

# %%
# Mengevaluasi model pada test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# %%
# plot akurasi
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=0.85, color='r', linestyle='--', label='Target Accuracy (85%)')
plt.axhline(y=0.96, color='g', linestyle='--', label='Threshold Accuracy (96%)')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# mendapatkan prediksi dan ground truth
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Membuat confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# %%
# laporan klasifikasi detail
print("\nLaporan Klasifikasi Detail:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# %% [markdown]
# ## Konversi Model

# %%
# Pastikan direktori yang diperlukan sudah ada
base_dir = '/content/drive/MyDrive/dicoding/pengembangan ML/submission 2'
os.makedirs(f'{base_dir}/saved_model', exist_ok=True)
os.makedirs(f'{base_dir}/tfjs_model', exist_ok=True)
os.makedirs(f'{base_dir}/tflite', exist_ok=True)

# %%
# 1. Menyimpan model dalam format SavedModel
tf.saved_model.save(model, f'{base_dir}/saved_model')
print("Model telah disimpan dalam format SavedModel.")

# 2. Menyimpan model dalam format Tensorflow.js
tfjs.converters.save_keras_model(model, f'{base_dir}/tfjs_model')
print("Model telah disimpan dalam format Tensorflow.js.")

# 3. Menyimpan model dalam format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(f'{base_dir}/tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)

with open(f'{base_dir}/tflite/label.txt', 'w') as f:
    for label in class_names:
        f.write(f"{label}\n")

print("Model telah disimpan dalam format Tensorflow Lite.")

# %% [markdown]
# ## Inference (Optional)

# %%
from keras.layers import TFSMLayer

# Menggunakan TFSMLayer untuk inferensi dari SavedModel
model = TFSMLayer(f'{base_dir}/saved_model', call_endpoint='serving_default')

# Fungsi preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Daftar path gambar yang ingin diuji
test_image_paths = [
    '/content/drive/MyDrive/dicoding/pengembangan ML/submission 2/animals_dataset_split/test/chicken/chicken (89).jpeg',
    '/content/drive/MyDrive/dicoding/pengembangan ML/submission 2/animals_dataset_split/test/dog/dog (286).jpeg',
    '/content/drive/MyDrive/dicoding/pengembangan ML/submission 2/animals_dataset_split/test/horse/horse (78).jpeg'
]

# Melakukan inferensi menggunakan model SavedModel
plt.figure(figsize=(20, 10))
for i, img_path in enumerate(test_image_paths):
    img = preprocess_image(img_path)
    prediction = model(img)  # Pakai model TFSMLayer langsung
    prediction = prediction['output_0']
    prediction = prediction.numpy()

    # Prediksi
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Menampilkan gambar dan hasil prediksi
    original_img = plt.imread(img_path)
    plt.subplot(1, len(test_image_paths), i + 1)
    plt.imshow(original_img)
    true_class = img_path.split('/')[-2]
    plt.title(f"True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.2f}%")
    plt.axis('off')

plt.tight_layout()
plt.savefig(f'{base_dir}/savedmodel_inference.png')
plt.show()


