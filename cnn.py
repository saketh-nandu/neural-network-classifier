import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class CNNImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the CNN Image Classifier
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes for classification
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def preprocess_image(self, image_path):
        """
        Preprocess a single image using OpenCV
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def augment_image(self, image):
        """
        Apply data augmentation using OpenCV
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image
        """
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            image = np.clip(image, 0, 1)
        
        # Random blur
        if np.random.random() > 0.7:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def load_and_preprocess_data(self, data_dir, validation_split=0.2):
        """
        Load and preprocess dataset
        
        Args:
            data_dir: Directory containing class subdirectories
            validation_split: Fraction of data to use for validation
            
        Returns:
            Preprocessed training and validation data
        """
        # Use tf.keras.utils.image_dataset_from_directory for efficient loading
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32
        )
        
        # Normalize pixel values
        normalization_layer = layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def build_model(self):
        """
        Build the CNN model architecture
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation layer
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_ds, val_ds, epochs=50):
        """
        Train the CNN model
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, test_ds):
        """
        Evaluate the model on test dataset
        
        Args:
            test_ds: Test dataset
            
        Returns:
            Test loss and accuracy
        """
        test_loss, test_accuracy = self.model.evaluate(test_ds, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_image(self, image_path, class_names):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to the image
            class_names: List of class names
            
        Returns:
            Predicted class and confidence
        """
        # Preprocess the image
        image = self.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return class_names[predicted_class], confidence
    
    def generate_confusion_matrix(self, test_ds, class_names):
        """
        Generate and plot confusion matrix
        
        Args:
            test_ds: Test dataset
            class_names: List of class names
        """
        # Get predictions
        y_pred = []
        y_true = []
        
        for images, labels in test_ds:
            predictions = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage
def main():
    """
    Main function to demonstrate the CNN classifier
    """
    # Initialize the classifier
    classifier = CNNImageClassifier(input_shape=(224, 224, 3), num_classes=10)
    
    # Build the model
    model = classifier.build_model()
    print("Model Architecture:")
    model.summary()
    
    # Example: Load data (replace with your actual data directory)
    # train_ds, val_ds = classifier.load_and_preprocess_data('path/to/your/dataset')
    
    # Example: Train the model
    # history = classifier.train_model(train_ds, val_ds, epochs=50)
    
    # Example: Plot training history
    # classifier.plot_training_history()
    
    # Example: Evaluate the model
    # test_loss, test_accuracy = classifier.evaluate_model(test_ds)
    
    # Example: Generate confusion matrix
    # class_names = ['class1', 'class2', 'class3', ...]  # Replace with actual class names
    # classifier.generate_confusion_matrix(test_ds, class_names)
    
    # Example: Make predictions
    # predicted_class, confidence = classifier.predict_image('path/to/image.jpg', class_names)
    # print(f"Predicted: {predicted_class} (Confidence: {confidence:.4f})")
    
    # Example: Save the model
    # classifier.save_model('cnn_classifier.h5')
    
    print("CNN Image Classifier implementation complete!")

if __name__ == "__main__":
    main()

# Additional utility functions
def create_sample_data():
    """
    Create sample synthetic data for testing
    """
    # Generate synthetic image data
    X = np.random.rand(1000, 224, 224, 3)
    y = np.random.randint(0, 10, 1000)
    
    return X, y

def plot_sample_images(dataset, class_names, num_images=9):
    """
    Plot sample images from the dataset
    
    Args:
        dataset: TensorFlow dataset
        class_names: List of class names
        num_images: Number of images to display
    """
    plt.figure(figsize=(10, 10))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Class: {class_names[labels[i]]}")
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def calculate_model_size(model):
    """
    Calculate the size of the model
    
    Args:
        model: Keras model
        
    Returns:
        Model size in MB
    """
    param_count = model.count_params()
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    
    print(f"Model Parameters: {param_count:,}")
    print(f"Estimated Model Size: {model_size_mb:.2f} MB")
    
    return model_size_mb