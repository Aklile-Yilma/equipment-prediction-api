
# Equipment Failure Prediction API

AI-powered equipment failure prediction service deployed on Render.

## 🚀 Features

- Real-time equipment failure prediction
- Batch processing for multiple equipment
- Risk level assessment (High/Medium/Low)
- Maintenance recommendations
- RESTful API with JSON responses

## 📡 API Endpoints

### POST /predict
Predict failure for single equipment.

**Request:**
```json
{
  "id": "DEF_668288",
  "type": "Defibrillator",
  "manufacturer": "Medtronic",
  "modelType": "LIFEPAK 20e",
  "operatingHours": 23,
  "installationDate": "2025-05-30T00:00:00.000Z"
}
```

**Response:**
```json
{
  "equipment_id": "DEF_668288",
  "prediction": {
    "days_to_failure": 45,
    "predicted_failure_date": "2025-07-26",
    "failure_type": "Battery Failure",
    "risk_level": "Medium",
    "confidence": "87.5%"
  },
  "recommendations": [
    "⚠️ Schedule maintenance within 14 days",
    "🔋 Check battery voltage and capacity"
  ]
}
```

### POST /predict-batch
Process multiple equipment at once.

### GET /health
Health check endpoint.

## 🛠️ Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set model path:
   ```bash
   export MODEL_PATH=models/equipment_model.pkl
   ```

3. Run the app:
   ```bash
   python app.py
   ```

## 🚀 Deployment

This API is configured for automatic deployment on Render.

## 📊 Model Information

- **Algorithm:** XGBoost + Random Forest Ensemble
- **Features:** Equipment age, operating hours, maintenance history, etc.
- **Accuracy:** ~90% for failure type prediction
- **Training Data:** Hospital equipment maintenance records

## 🔐 Security

- Input validation on all endpoints
- Error handling and logging
- Rate limiting recommended for production

## 📞 Support

For issues or questions, please check the API documentation at the root endpoint.
