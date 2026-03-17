#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <arduinoFFT.h>
#include "gesture_recognition.h"

constexpr int WINDOW_SIZE = 60;     // 60 samples window size (~0.5 seconds considering the IMU rate of 119hz)
constexpr int STEP_SIZE = 30;       // 50% overlap
constexpr int NUM_AXES = 6;         // aX, aY, aZ, gX, gY, gZ
constexpr int FFT_SAMPLES = 64;     // Nearest power of 2 for 60
constexpr float SAMPLING_FREQ = 119.0;

float window_data[NUM_AXES][WINDOW_SIZE];
int samplesRead = 0;

const float accelerationThreshold = 2.0;
const float PREDICTION_THRESHOLD = 0.70;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

double vReal[FFT_SAMPLES];
double vImag[FFT_SAMPLES];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_SAMPLES, SAMPLING_FREQ);


void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddReshape();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model Info");
  
  // Check expected input features
  Serial.print("Input Size is ");
  Serial.println(input->dims->data[input->dims->size - 1]);

  Serial.print("Input dtype is ");
  if (input->type == kTfLiteFloat32) {
    Serial.println("f32");
  } else if (input->type == kTfLiteInt8) {
    Serial.println("int8");
  } else {
    Serial.print("with id: ");
    Serial.println(input->type);
  }
    
  Serial.println("System Ready. Collecting sliding windows...");
}

void extract_features_and_predict() {
  float features[NUM_FEATURES]; // dynamically sized via header (in case I want to add some features later)
  int feature_idx = 0;

  // need to store means and standard deviations to calculate Correlation later
  float axis_means[NUM_AXES];
  float axis_std_devs[NUM_AXES];

  for (int axis = 0; axis < NUM_AXES; axis++) {
    float sum = 0, sq_sum = 0;
    float min_val = window_data[axis][0];
    float max_val = window_data[axis][0];
    
    // Mean, Min, Max
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float val = window_data[axis][i];
      sum += val;
      sq_sum += (val * val);
      if (val < min_val) min_val = val;
      if (val > max_val) max_val = val;
    }
    float mean = sum / WINDOW_SIZE;
    
    // Std Dev & RMS
    float variance_sum = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      variance_sum += pow(window_data[axis][i] - mean, 2);
    }
  float std_dev = sqrt(fmax(0.0, variance_sum / WINDOW_SIZE));
  float rms = sqrt(fmax(0.0, sq_sum / WINDOW_SIZE));

    // save for correlation math
    axis_means[axis] = mean;
    axis_std_devs[axis] = std_dev;

    // PSD
    for (int i = 0; i < FFT_SAMPLES; i++) {
      vReal[i] = (i < WINDOW_SIZE) ? window_data[axis][i] : 0.0;
      vImag[i] = 0.0;
    }
    
    FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();

    double max_psd = 0;
    for (int i = 1; i < (FFT_SAMPLES / 2); i++) {
      double power = (vReal[i] * vReal[i]) / WINDOW_SIZE; 
      if (power > max_psd) max_psd = power;
    }

    float raw_features[6] = {mean, std_dev, rms, min_val, max_val, (float)max_psd};
    
    // normalize features
    for(int j = 0; j < 6; j++) {
      features[feature_idx] = (raw_features[j] - scaler_mean[feature_idx]) / scaler_scale[feature_idx];
      feature_idx++;
    }
  }

  
  // compute correlation for circular motions
  float cov_aXY = 0;
  for(int i = 0; i < WINDOW_SIZE; i++) {
    cov_aXY += (window_data[0][i] - axis_means[0]) * (window_data[1][i] - axis_means[1]);
  }
  cov_aXY /= WINDOW_SIZE;
  float corr_aXY = (axis_std_devs[0] > 0 && axis_std_devs[1] > 0) ? (cov_aXY / (axis_std_devs[0] * axis_std_devs[1])) : 0.0;

  float cov_gXY = 0;
  for(int i = 0; i < WINDOW_SIZE; i++) {
    cov_gXY += (window_data[3][i] - axis_means[3]) * (window_data[4][i] - axis_means[4]);
  }
  cov_gXY /= WINDOW_SIZE;
  float corr_gXY = (axis_std_devs[3] > 0 && axis_std_devs[4] > 0) ? (cov_gXY / (axis_std_devs[3] * axis_std_devs[4])) : 0.0;

  // normalize and append the two new correlation features
  features[feature_idx] = (corr_aXY - scaler_mean[feature_idx]) / scaler_scale[feature_idx];
  feature_idx++;
  features[feature_idx] = (corr_gXY - scaler_mean[feature_idx]) / scaler_scale[feature_idx];
  feature_idx++;


  // copy into model input buffer
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features[i];
  }

  // perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  // output prediction
  float max_prob = 0.0;
  int best_match_index = -1;

  for (int i = 0; i < NUM_CLASSES; i++) {
    // for debug: print all classes and associated confidence
    // Serial.print(GESTURE_LABELS[i]);
    // Serial.print(" with confidence: ");
    // Serial.println(output->data.f[i]);
    if (output->data.f[i] > max_prob) {
      max_prob = output->data.f[i];
      best_match_index = i;
    }
  }

  if (best_match_index != -1 && max_prob >= PREDICTION_THRESHOLD) {
    Serial.print(GESTURE_LABELS[best_match_index]);
    Serial.print(" ("); 
    Serial.print(max_prob * 100, 0); 
    Serial.println("%)");
  }
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);
    IMU.readGyroscope(gX, gY, gZ);

    window_data[0][samplesRead] = aX;
    window_data[1][samplesRead] = aY;
    window_data[2][samplesRead] = aZ;
    window_data[3][samplesRead] = gX;
    window_data[4][samplesRead] = gY;
    window_data[5][samplesRead] = gZ;

    samplesRead++;

    if (samplesRead == WINDOW_SIZE) {
      bool isMoving = false;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        float aSum = fabs(window_data[0][i]) + fabs(window_data[1][i]) + fabs(window_data[2][i]);
        if (aSum >= accelerationThreshold) {
          isMoving = true;
          break;
        }
      }

      // Only perform inference if device is moving,
      // this is effectively better than an 'idle' class
      if (isMoving) {
        extract_features_and_predict();
      }

      for (int axis = 0; axis < NUM_AXES; axis++) {
        for (int i = 0; i < (WINDOW_SIZE - STEP_SIZE); i++) {
          window_data[axis][i] = window_data[axis][i + STEP_SIZE];
        }
      }
      
      samplesRead = WINDOW_SIZE - STEP_SIZE; 
    }
  }
}