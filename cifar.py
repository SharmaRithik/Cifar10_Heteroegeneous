import numpy as np
import tensorflow as tf
import os

def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15)

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

def make_prediction(model, image):
    prediction = model.predict(np.array([image]))
    predicted_class = np.argmax(prediction)
    return predicted_class

def custom_prediction(model, image):
    # Preprocess the image as needed (e.g., reshaping, normalization)
    image = np.array([image])  # Add batch dimension

    # Perform manual forward pass
    prediction = manual_forward_pass(model, image)

    # For classification, return the class with the highest probability
    custom_predicted_class = np.argmax(prediction, axis=-1)
    return custom_predicted_class

def manual_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def manual_forward_pass(model, input_data):
    a = input_data
    for layer in model.layers:
        if 'flatten' in layer.name:
            a = a.reshape(a.shape[0], -1)
        elif 'dense' in layer.name:
            W, b = layer.get_weights()
            a = np.dot(a, W) + b
            if 'softmax' in layer.get_config()['activation']:
                a = manual_softmax(a)
            else:
                a = layer.activation(a)
    return a

def save_matrices(layer_name, a, W, b):
    if not os.path.exists('saved_matrices'):
        os.makedirs('saved_matrices')
    np.save(f'saved_matrices/{layer_name}_a.npy', a)
    np.save(f'saved_matrices/{layer_name}_w.npy', W)
    np.save(f'saved_matrices/{layer_name}_b.npy', b)
    print(f"Matrices saved for layer: {layer_name}")

def display_matrices_during_forward_pass(model, image):
    a = np.array([image])
    for layer in model.layers:
        if 'flatten' in layer.name:
            a = a.reshape(a.shape[0], -1)
        elif 'dense' in layer.name:
            W, b = layer.get_weights()
            z = np.dot(a, W) + b
            save_matrices(layer.name, a, W, b)
            print(f"\nLayer: {layer.name}")
            print("Weights shape:", W.shape)
            print("Biases shape:", b.shape)
            print("Input to the layer (a):", a.shape)
            print("Matrix multiplication result (z):", z.shape)
            if 'softmax' in layer.get_config()['activation']:
                a = manual_softmax(z)
            else:
                a = layer.activation(z)
            print("Output of the layer (after activation):", a.shape)
        else:
            print(f"Skipping non-dense layer: {layer.name}")

# Main execution
x_train, y_train, x_test, y_test = load_cifar10_dataset()
model = create_model()
train_model(model, x_train, y_train)
evaluate_model(model, x_test, y_test)

first_image = x_test[0]
predicted_class = make_prediction(model, first_image)
print("\nPredicted Class:", predicted_class)
print("True Class:", y_test[0][0])

# Custom prediction function
custom_predicted_class = custom_prediction(model, first_image)
print("Custom Predicted Class:", custom_predicted_class[0])

display_matrices_during_forward_pass(model, first_image)
