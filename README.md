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

Then, you have two options for the actual training: first, you can use the Jupiter Notebook at this [link](https://drive.google.com/file/d/1X-pUrdSQqot0ESKmBRob8OxhTnZvcxDt/view?usp=drive_link). If you do, make sure to follow the instructions at the top of the page to upload the training CSV data for your classes.

If you prefer, you can just use the two python scripts `extract_features.py` and `train_model.py` (in this order of course). If you decide to go this route, you need to be on a python version supported by tensorflow, create some kind of virtual environment, install the `requirements.txt` with pip and then run the scripts. Make sure to `cd` into the training [folder](training) before running them, so that file paths are correctly interpreted.

Either method will produce a `features.csv` file as intermediary step, and then directly export the model to a C++ header file `gesture_recognition.h`, which will be used in the next step.

If you use the Jupiter Notebook, you can download the header file and place it in the correct [folder](inference), otherwise it will be generated already there.

## 3. Inference

Once you have your C++ header file in the inference [folder](inference), you can open the folder as an Arduino Sketch, compile it and load it to the device with the exact same procedure as before. Tuning into the serial output of the board will allow you to see inference for every detected movement. 