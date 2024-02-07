# Image Similarity Calculations by Euclidean Similarity

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 모델 로드
model = tf.keras.models.load_model('Model.h5')

# 이미지 불러오기 및 전처리
image_path_1 = '/path/to/image.PNG'
image_path_2 = '/path/to/image.PNG'

img1 = tf.keras.preprocessing.image.load_img(image_path_1, color_mode='grayscale', target_size=(28, 28))
img1 = tf.keras.preprocessing.image.img_to_array(img1)
img1 = img1 / 255.0  # 모델에 입력하기 전에 정규화

img2 = tf.keras.preprocessing.image.load_img(image_path_2, color_mode='grayscale', target_size=(28, 28))
img2 = tf.keras.preprocessing.image.img_to_array(img2)
img2 = img2 / 255.0

# 모델 예측
embedding_1 = model.predict(np.expand_dims(img1, axis=0))
embedding_2 = model.predict(np.expand_dims(img2, axis=0))

# 유클리디안 거리 계산
euclidean_distance = np.linalg.norm(embedding_1 - embedding_2)

# 이미지 표시
plt.subplot(1, 2, 1)
plt.imshow(img1[:, :, 0])  # 첫 번째 이미지 표시
plt.title('img1')

plt.subplot(1, 2, 2)
plt.imshow(img2[:, :, 0])  # 두 번째 이미지 표시
plt.title('img2')

plt.suptitle(f"Euclidean Distance: {euclidean_distance:.4f}", fontsize=14)
plt.show()

print(euclidean_distance)