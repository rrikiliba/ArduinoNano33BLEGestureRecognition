import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import textwrap

INPUT_FILE = 'features.csv'
HEADER_FILE = '../inference/gesture_recognition.h'

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Save the original class names to export later
    gesture_names = label_encoder.classes_

    # Split into Features (X) and Labels (Y)
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    num_classes = len(np.unique(Y))

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # --- NORMALIZATION ---
    # This is required by the assignment. We scale data to mean=0, std=1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- MODEL TRAINING ---
    print("\nTraining Neural Network...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

    # --- EVALUATION ---
    print("\nEvaluating Model on Test Data...")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- EXPORT TO TFLITE ---
    print("\nConverting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimize for size (crucial for microcontrollers)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # --- GENERATE C++ HEADER FILE (.h) ---
    print(f"\nGenerating C++ Header: {HEADER_FILE}...")

    # 1. Convert TFLite binary to Hex String array
    hex_array = [format(val, '#04x') for val in tflite_model]
    hex_array_str = ', '.join(hex_array)
    wrapped_hex_array = textwrap.fill(hex_array_str, width=100)

    # 2. Extract Scaler variables to format as C++ arrays
    scaler_means_str = ', '.join([str(val) for val in scaler.mean_])
    scaler_scales_str = ', '.join([str(val) for val in scaler.scale_])
    
    # Format the gesture labels for C++
    labels_str = ', '.join([f'"{name}"' for name in gesture_names])

    # 3. Write everything into a single header file
    c_code = f"""// Automatically generated model and scaler data
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// The number of classes
const int NUM_CLASSES = {len(gesture_names)};

// The gesture labels
const char* const GESTURE_LABELS[] = {{
    {labels_str}
}};

// The number of features extracted per window
const int NUM_FEATURES = {X_train.shape[1]};

// Normalization Means
const float scaler_mean[{X_train.shape[1]}] = {{
    {scaler_means_str}
}};

// Normalization Scales (Standard Deviations)
const float scaler_scale[{X_train.shape[1]}] = {{
    {scaler_scales_str}
}};

// TFLite Model Array
const unsigned int g_model_len = {len(tflite_model)};
alignas(8) const unsigned char g_model[] = {{
{wrapped_hex_array}
}};

#endif // MODEL_DATA_H
"""

    with open(HEADER_FILE, 'w') as f:
        f.write(c_code)

    print(f"Success! You can now include {HEADER_FILE} in your Arduino Sketch.")

if __name__ == "__main__":
    main()