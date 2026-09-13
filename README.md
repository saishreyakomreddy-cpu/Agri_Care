# AgriVision Grow

AgriVision Grow is a Streamlit agriculture assistant for crop recommendation, plant disease detection, weather insights, fertilizer guidance, irrigation planning, and prediction history.

## Run the app

```powershell
cd "D:\agri sheid\AgriVision-Grow"
streamlit run app.py
```

## Project data

- `data/crop_data.csv` contains the crop recommendation examples.
- `data/disease_data/` contains the training and validation class folders.
- `models/train_crop_model.py` trains the crop recommendation model.
- `models/train_disease_model.py` trains the MobileNetV2 disease model after images are added.
- `database/agrivision.db` stores users and prediction history.

The disease model artifact is generated only after the disease dataset is populated and the TensorFlow training script completes.
