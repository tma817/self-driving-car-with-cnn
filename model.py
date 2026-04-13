import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import cv2 as cv
import random 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load data procedure
df = pd.read_csv('data/driving_log.csv')
df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

# Set max samples per bin
max_samples = 300
samples_per_bin, bins = np.histogram(df['steering'], bins=25)

# Remove excess samples from overrepresented bins
remove_list = []
for i in range(25):
    list_ = []
    for j in range(len(df['steering'])):
        if df['steering'][j] >= bins[i] and df['steering'][j] <= bins[i+1]:
            list_.append(j)
    list_ = shuffle(list_)
    list_ = list_[max_samples:]
    remove_list.extend(list_)

df.drop(df.index[remove_list], inplace=True)

# Plot histogram
plt.hist(df['steering'], bins=25, label='steering data')
plt.grid(axis='y', linestyle='--')
plt.legend()
plt.title('Balanced Steering Angle Distribution')
plt.xlabel('Steering Angle')
plt.ylabel('Count')
plt.show()
print(f'Remaining samples: {len(df)}')

# --- Preprocessing ---
def preProcessing(img):
    img = img[60:135, :, :]
    img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.resize(img, (200, 66))
    img = img / 255
    return img

# --- Data Augmentation ---
def augmentFlip(img, steering):
    img = cv.flip(img, 1)
    steering = steering * -1
    return img, steering

def augmentBrightness(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = np.array(img, dtype=float)
    factor = random.uniform(0.2, 1.2)
    img[:,:,2] = img[:,:,2] * factor
    img[:,:,2][img[:,:,2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return img

def augmentZoom(img):
    zoom = random.uniform(1.0, 1.3)
    h, w = img.shape[:2]
    new_h = int(h / zoom)
    new_w = int(w / zoom)
    y1 = int((h - new_h) / 2)
    x1 = int((w - new_w) / 2)
    img = img[y1:y1+new_h, x1:x1+new_w]
    img = cv.resize(img, (w, h))
    return img

def augmentImage(img, steering):
    if random.random() < 0.5:
        img, steering = augmentFlip(img, steering)

    if random.random() < 0.5:
        img = augmentBrightness(img)

    if random.random() < 0.3:
        img = augmentZoom(img)

    return img, steering

# --- Model ---
def createModel():
    model = Sequential()
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='elu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model

# --- Batch Generator ---
def batchGenerator(X, y, batch_size=32, training=False):
    while True:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            idx = random.randint(0, len(X)-1)
            img = cv.imread(X[idx].strip())
            steering = y[idx]
            if training:
                img, steering = augmentImage(img, steering)
            img = preProcessing(img)
            batch_X.append(img)
            batch_y.append(steering)
        yield np.array(batch_X), np.array(batch_y)

# --- Data Preparation ---
X = df['center'].values
y = df['steering'].values

X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training samples:', len(X_train))
print('Test samples:', len(X_test))

# Visualize sample images
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax in axes.flat:
    idx = random.randint(0, len(X)-1)
    img = cv.imread(X[idx].strip())
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax.set_title(f'steering: {y[idx]:.3f}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# --- Training ---
model = createModel()
model.summary()

early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           verbose=1)

history = model.fit(
    batchGenerator(X_train, y_train, 32, training=True),
    steps_per_epoch=150,   
    epochs=50,
    validation_data=batchGenerator(X_test, y_test, 32, training=False),
    validation_steps=100, 
    callbacks=[early_stop]
)

model.save('model.h5')
print('Model saved!')

# --- Plot Training Loss ---
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --- Data Check ---
df_check = pd.read_csv('data/driving_log.csv')
df_check.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
print(f'Total samples: {len(df_check)}')
print(f'Left turns:    {len(df_check[df_check["steering"] < 0])}')
print(f'Straight:      {len(df_check[df_check["steering"] == 0])}')
print(f'Right turns:   {len(df_check[df_check["steering"] > 0])}')
print(f'Mean steering: {df_check["steering"].mean():.4f}')