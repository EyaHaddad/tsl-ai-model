# tsl-ai

TensorFlow/TFLite hand-sign recognition workflow with MediaPipe real-time inference.

## Project Overview

- `data_preprocessing.ipynb`: dataset preparation pipeline.
- `lstm.ipynb`: baseline LSTM model training and export.
- `lstm-cnn.ipynb`: hybrid LSTM-CNN architecture training and export.
- `real_time_mediapipe.ipynb`: webcam inference with MediaPipe + TFLite.

## Environment Setup

Prerequisites:

- Python 3.10+
- `pip`

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected Workspace Structure

```text
tsl-ai/
├── data_preprocessing.ipynb        # dataset preparation
├── lstm-cnn.ipynb                  # main TensorFlow training/export notebook
├── lstm.ipynb                      # baseline model experiments
├── real_time_mediapipe.ipynb       # webcam inference pipeline
├── requirements.txt
├── artifacts/
│   ├── lstm_dataset.npz
│   ├── lstm_dataset_meta.json
│   └── tf_model_dir/
│       ├── baseline/
│       │   ├── model_float16.tflite
│       │   └── model_float32.tflite
│       └── lstmhybrid_tf/
│           ├── model_builtin_float32.tflite
│           ├── model_float16.tflite
│           └── model_float32.tflite
└── mp/
    └── hand_landmarker.task
```

## Training and Export Flow

Run `lstm-cnn.ipynb` end-to-end to:

1. Load dataset artifacts.
2. Expand sequences with frame duplication (target sequence length = 20).
3. Train the TensorFlow hybrid LSTM model.
4. Export Keras/SavedModel and TFLite variants.
5. Validate inference on generated TFLite output.

## Real-Time Inference

Run `real_time_mediapipe.ipynb` to test webcam inference.

- Baseline path uses `artifacts/tf_model_dir/baseline/model_float32.tflite`.
- Hybrid path should prefer `artifacts/tf_model_dir/lstmhybrid_tf/model_builtin_float32.tflite`.

## Notes and Troubleshooting

- Training notebooks were executed with a Colab kernel originally, so some cells may reference Drive-style paths.
- If a TFLite model fails with Flex op errors, use `model_builtin_float32.tflite` from `artifacts/tf_model_dir/lstmhybrid_tf`.
- `tf.lite.Interpreter` deprecation warnings can appear on newer TF versions; current notebook logic remains compatible with existing setup.
