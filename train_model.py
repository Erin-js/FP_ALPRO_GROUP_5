import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
import json
import re

DATA_DIR = 'dataset'
MODEL_DIR = 'models'
IMG_SHAPE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def extract_brand_names(real_folder_path):
    brands = set()
    
    if not os.path.exists(real_folder_path):
        return []

    for filename in os.listdir(real_folder_path):
        name = os.path.splitext(filename)[0].lower()
        clean_name = re.sub(r'[^a-z]', '', name)
        
        if len(clean_name) > 2:
            brands.add(clean_name)
    
    sorted_brands = sorted(list(brands))
    return sorted_brands

def train():

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    print(f" Mapping Kelas: {train_generator.class_indices}")

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE+(3,))

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining Tahap 1 (Basic Learning)...")
    model.fit(train_generator, epochs=5, validation_data=val_generator)

    print("\nTraining Tahap 2 (Fine Tuning - Mencerdaskan Model)...")
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_generator, epochs=10, validation_data=val_generator)

    print("\n Menyimpan Model & Database...")
    
    model.save(os.path.join(MODEL_DIR, 'logo_resnet_model.h5'))
    
    path_output_asli = os.path.join(DATA_DIR, 'output')
    brand_list = extract_brand_names(path_output_asli)
    
    with open(os.path.join(MODEL_DIR, 'brands_db.json'), 'w') as f:
        json.dump(brand_list, f)

if __name__ == "__main__":
    train()