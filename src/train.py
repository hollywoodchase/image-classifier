import tensorflow as tf
from src.data_loader import load_cifar10_data
from src.model import build_cnn_model
import matplotlib.pyplot as plt

def train_model():
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = load_cifar10_data()

    # Build the model
    model = build_cnn_model((32, 32, 3), 10)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    # Save the model in the recommended format
    model.save('models/cnn_model.keras')
    
    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_acc}')
    return history

if __name__ == '__main__':
    train_model()
