# 커스텀 이미지 데이터를 One-Hot Encoding 형태로 변환시켜주는 코드

from keras.utils import to_categorical
import numpy as np

# 가정: custom_labels가 이미 정수 형태로 되어 있다고 가정
# 예를 들어, [0, 1, 2, 3, ..., 9] 형태로 레이블이 있다면,
custom_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 확인을 위해 출력
print("Before Conversion:", custom_labels)

# custom_labels를 numpy 배열로 변환
custom_labels = np.array(custom_labels)

# 정수 형태로 변환된 것을 확인
print("After Conversion:", custom_labels)
