import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 모델 로드
model = tf.keras.models.load_model('Model.h5')


# 이미지 불러오기 및 전처리
image_path_1 = '/Users/seunghunjang/Desktop/Coordination/Coordination/train/TOP/TOP_1.jpg'
image_path_2 = '/Users/seunghunjang/Desktop/Coordination/Coordination/train/TOP/TOP_5.jpg'

img1 = tf.keras.preprocessing.image.load_img(image_path_1, color_mode='grayscale',target_size=(28, 28))
img1 = tf.keras.preprocessing.image.img_to_array(img1)
img1 = tf.expand_dims(img1, axis=0)
img1 = img1 / 255.0  # 모델에 입력하기 전에 정규화

img2 = tf.keras.preprocessing.image.load_img(image_path_2, color_mode='grayscale', target_size=(28, 28))
img2 = tf.keras.preprocessing.image.img_to_array(img2)
img2 = tf.expand_dims(img2, axis=0)
img2 = img2 / 255.0

# 모델 예측
embedding_1 = model.predict(img1)
embedding_2 = model.predict(img2)

# 유사도 계산 (코사인 유사도 사용)
similarity = cosine_similarity(embedding_1.reshape(1, -1), embedding_2.reshape(1, -1))

print(f"Cosine Similarity: {similarity[0][0]}")
