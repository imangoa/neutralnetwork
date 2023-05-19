
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ProgbarLogger

# 定义数据集路径和类别
data_path = '/mnt/data'
categories = ['cheku', 'suidao']

# 加载数据集
data = []
label = []
for category in categories:
    path = os.path.join(data_path, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))
        data.append(image)
        label.append(categories.index(category))

# 将数据和标签分割为训练集和测试集
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)

# 将训练集和测试集转换成 Numpy 数组
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

# 将标签转换为独热编码格式
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# 创建卷积神经网络模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 定义回调函数，显示训练进度
# logger = ProgbarLogger(count_mode='steps')
#
# # 训练模型
# model.fit(train_data, train_label, batch_size=32, epochs=20, validation_data=(test_data, test_label), callbacks=[logger])
# # 定义回调函数，显示训练进度
logger = ProgbarLogger(count_mode='steps')

# 训练模型
model.fit(train_data, train_label, epochs=20, validation_data=(test_data, test_label), callbacks=[logger], steps_per_epoch=len(train_data) // 32, validation_steps=len(test_data) // 32)
model.save('my_model.h5')

# 评估模型
score = model.evaluate(test_data, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 使用模型进行预测
img_path = './1.jpg'
image = cv2.imread(img_path)
image = cv2.resize(image, (128, 128))
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)

# 打印预测结果
print('Prediction:', categories[np.argmax(prediction)])

