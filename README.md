# Gesture Recognition on Arduino Nano 33 BLE

This represents the submission for the first assignment of the course "Low-power embedded systems" of students:

- Riccardo Libanora
- Jacopo Scanavacca

# Instructions

The repository contains the files necessary for all three steps of the TinyML pipeline to deploy a minimal model for gesture recognition using accelerometer and gyroscope data on Arduino Nano 33 BLE

## 1. Data collection

In [this](data_collection) folder is the arduino sketch used to gather the data used to train the model. It is simply the one provided by the professor and used during the course lectures, but we included it for the sake of completeness.

You just need to compile and load this script to the device via Arduino tools (such as Arduino IDE or Cloud), and structured CSV data will be printed to the serial output as the device detects movement.

To gather data, you can therefore continuosly make the desired movement or gesture, then copy the terminal output to your_gesture.csv and place it in the [dataset](training/dataset) folder. Note that the name given to the corresponding class will be the filename you choose, so decide accordingly.

## 2. Model training

This project is set up in a way that assumes you want to gather data on your own classes, hence why the previous step was reported here. However, we also made sure to upload the exact data we used for training in a separate [branch](https://github.com/rrikiliba/ArduinoNano33BLEGestureRecognition/tree/dataset), if that's what floats your boat.

The class name assignment, as already mentioned, is fully automated, so you can add whatever class you want by just creating its training data.

The actual model training is done entirely within this Jupiter notebook [file](training/training.ipynb), which explains the steps pretty well on its own. If you don't know how to open it, you can either:

- open it in VS Code using the official [extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- open [Google Colab](https://colab.research.google.com/), go to File > Open notebook, select the GitHub tab and paste this repo's link

Once executed all the cells, the script will produce a `features.csv` file as intermediary step, and then directly export the model to a C++ header file `gesture_recognition.h`, which will be used in the next step. A `.tflite` file is also generated, but you can ignore it.

Make sure to generally follow the instructions in the notebook and, once you obtain your `.h` file, place it in this [folder](inference).

## 3. Inference

Once you have your C++ header file in the inference [folder](inference), you can open the folder as an Arduino Sketch, compile it and load it to the device with the exact same procedure as before. Tuning into the serial output of the board will allow you to see inference for every detected movement. 