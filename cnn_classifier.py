"""
Cat vs Dog Image Classifier using Convolutional Neural Network
Author: Ahmad Hammam
Date: July-19-2025
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

def create_data_generators():
    """Create data generators for training and testing"""
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Test data generator (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training set
    training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )
    
    # Load test set
    test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )
    
    return training_set, test_set

def build_cnn_model():
    """Build and compile the CNN model"""
    # Initialize CNN
    cnn = tf.keras.models.Sequential()
    
    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(
        filters=32, 
        kernel_size=3, 
        activation='relu', 
        input_shape=[64, 64, 3]
    ))
    
    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())
    
    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    
    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # Compile the CNN
    cnn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return cnn

def train_model(model, training_set, test_set, epochs=25):
    """Train the CNN model"""
    print("Starting model training...")
    
    # Train the CNN
    history = model.fit(
        x=training_set,
        validation_data=test_set,
        epochs=epochs,
        verbose=1
    )
    
    return history

def save_model(model, filepath='models/cat_dog_classifier.h5'):
    """Save the trained model"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save(filepath)
    print(f"Model saved to {filepath}")

def plot_training_history(history):
    """Plot training accuracy and loss"""
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_single_image(model, image_path, class_indices):
    """Make prediction on a single image"""
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    # Make prediction
    result = model.predict(test_image)
    
    # Get class names
    class_names = {v: k for k, v in class_indices.items()}
    
    # Determine prediction
    prediction_prob = result[0][0]
    if prediction_prob > 0.5:
        prediction = class_names[1]  # Usually 'dogs'
        confidence = prediction_prob
    else:
        prediction = class_names[0]  # Usually 'cats'
        confidence = 1 - prediction_prob
    
    return prediction, confidence

def main():
    """Main function to run the complete pipeline"""
    print("Cat vs Dog Classifier Training Pipeline")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Create data generators
    print("\n1. Loading and preprocessing data...")
    training_set, test_set = create_data_generators()
    
    # Build model
    print("\n2. Building CNN model...")
    model = build_cnn_model()
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\n3. Training the model...")
    history = train_model(model, training_set, test_set)
    
    # Save model
    print("\n4. Saving the model...")
    save_model(model)
    
    # Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(history)
    
    # Test single predictions
    print("\n6. Testing single predictions...")
    
    # Test on sample images
    sample_images = [
        'dataset/single_prediction/cat_or_dog_1.jpg',
        'dataset/single_prediction/cat_or_dog_2.jpg'
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            prediction, confidence = predict_single_image(
                model, img_path, training_set.class_indices
            )
            print(f"Image: {img_path}")
            print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
            print("-" * 30)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()