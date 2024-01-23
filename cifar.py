import numpy as np
import tensorflow as tf

def load_cifar10_dataset():
    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
    return x_train, y_train, x_test, y_test

def create_model():
    # Define the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes in CIFAR-10
    ])
    return model

def train_model(model, x_train, y_train):
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Train the model
    model.fit(x_train, y_train, epochs=15)  # You can adjust the number of epochs

def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

def make_prediction(model, image):
    # Make a prediction
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
            a = a.reshape(a.shape[0], -1)  # Flatten the input for dense layers
        elif 'dense' in layer.name:
            W, b = layer.get_weights()
            a = np.dot(a, W) + b  # Matrix multiplication and bias addition
            if 'softmax' in layer.get_config()['activation']:
                a = manual_softmax(a)  # Apply softmax for output layer
            else:
                a = layer.activation(a)  # Apply other activation functions
        # Add handling for other layer types if necessary
    return a

def display_matrices_during_forward_pass(model, image):
    a = np.array([image])  # Reshape image to include batch dimension

    for layer in model.layers:
        if 'flatten' in layer.name:
            a = a.reshape(a.shape[0], -1)  # Flatten the image for dense layers
        elif 'dense' in layer.name:
            W, b = layer.get_weights()
            z = np.dot(a, W) + b  # Linear transformation

            print(f"\nLayer: {layer.name}")
            print("Weights shape:", W.shape)
            print("Biases shape:", b.shape)
            print("Input to the layer (a):", a.shape)
            print("Matrix multiplication result (z):", z.shape)

            if 'softmax' in layer.get_config()['activation']:
                a = manual_softmax(z)  # Apply manual softmax
            else:
                a = layer.activation(z)  # Other activations

            print("Output of the layer (after activation):", a.shape)
        else:
            print(f"Skipping non-dense layer: {layer.name}")

# Rest of your script...

# Main execution
x_train, y_train, x_test, y_test = load_cifar10_dataset()
model = create_model()
train_model(model, x_train, y_train)
evaluate_model(model, x_test, y_test)

first_image = x_test[0]
predicted_class = make_prediction(model, first_image)
print("\nPredicted Class:", predicted_class)
print("True Class:", y_test[0][0])

#custom prediction function
custom_predicted_class = custom_prediction(model, first_image)
print("Custom Predicted Class:", custom_predicted_class[0])


# Display matrices and intermediate results during forward pass
display_matrices_during_forward_pass(model, first_image)

