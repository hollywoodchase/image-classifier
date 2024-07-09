import tensorflow as tf
from src.data_loader import load_cifar10_data
from src.model import build_cnn_model

(train_images, train_labels), (test_images, test_labels) = load_cifar10_data()

model = build_cnn_model((32, 32, 3), 10)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
model.save('models/cnn_model.h5')
