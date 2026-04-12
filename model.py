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

max_samples = 400
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
    img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.resize(img, (200, 66))
    img = img / 255
    return img

