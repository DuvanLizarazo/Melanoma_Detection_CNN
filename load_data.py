import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Check if a saved model exists
if os.path.exists('melanoma_classifier.h5'):
    # Load the saved model if it exists
    model = load_model('melanoma_classifier.h5')
    print("Model loaded.")
else:
    # Data directories
    train_dir = 'skin-lesions/train'
    valid_dir = 'skin-lesions/valid'
    test_dir = 'skin-lesions/test'

    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Normalization for validation and test sets
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Validation data generator
    valid_generator = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Test data generator
    test_generator = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Define the model with a deeper architecture and regularization techniques
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    # Compile the model with L2 regularization
    from tensorflow.keras import regularizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model created.")

    # Define a checkpoint to save the model after each epoch
    checkpoint = ModelCheckpoint('melanoma_classifier_epoch_{epoch:02d}.h5', save_weights_only=False)

    # Train the model with more epochs and batch size
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=25,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        callbacks=[checkpoint]  # Pass the checkpoint callback to save the model after each epoch
    )

    # Save the final trained model
    model.save('melanoma_classifier_final.h5')
    print("Final model saved.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_accuracy}')
