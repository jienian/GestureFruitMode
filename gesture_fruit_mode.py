import os
from PIL import Image

# 创建主目录
DATA_PATH = 'Fruit_Data'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

fruits = ['Pear', 'Grape', 'Banana', 'Apple', 'Orange']

# 为每个水果类别创建子目录
for fruit in fruits:
    fruit_path = os.path.join(DATA_PATH, fruit)
    if not os.path.exists(fruit_path):
        os.makedirs(fruit_path)

uploaded_image_paths   = [
    '/content/fruit_1.pic.jpg',
    '/content/fruit_2.pic.jpg',
    '/content/fruit_3.pic.jpg',
    '/content/fruit_4.pic.jpg',
    '/content/fruit_5.pic.jpg'
]
for i, fruit in enumerate(fruits):
    image = Image.open(uploaded_image_paths[i])
    image = image.resize((64, 64))  # 调整大小
    fruit_path = os.path.join(DATA_PATH, fruit)
    image.save(os.path.join(fruit_path, f'{fruit}_{i}_pic.jpg'))

print("数据准备完成，目录结构已创建。")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 图像数据生成器
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 训练数据生成器
train_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

# 验证数据生成器
validation_generator = datagen.flow_from_directory(
    DATA_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

print("数据生成器创建成功。")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(fruits), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# 评估模型
# loss, accuracy = model.evaluate(validation_generator)
loss, accuracy = model.evaluate(train_generator)
print(f"Test Accuracy: {accuracy}")

# 转换为TensorFlow Lite模型
# 模型转换为TensorFlow Lite格式
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存 TensorFlow Lite 模型
with open('fruit_model.tflite', 'wb') as f:
    f.write(tflite_model)
