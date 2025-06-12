
#!/usr/bin/env python3
"""
Equipment Failure Prediction API for Render Deployment
"""

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import warnings

app = Flask(__name__)

# Global variables
trained_model = None
predictor = None

# CELL 2: Import Libraries and Main Model Class
# ================================================================================================

class EnhancedEquipmentFailurePredictionModel:
    """
    Enhanced model class - MUST match your training script exactly
    This allows proper unpickling of your saved model
    """
    def __init__(self, use_ensemble=True):
        # XGBoost Models (Primary)
        self.xgb_time_model = None
        self.xgb_type_model = None
        
        # Random Forest Models (Secondary/Ensemble)
        self.rf_time_model = None
        self.rf_type_model = None
        
        # Ensemble settings
        self.use_ensemble = use_ensemble
        
        # Preprocessing
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        categorical_cols = ['type', 'manufacturer', 'model', 'location', 'most_common_issue']
        
        df_encoded = df.copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    # Create simple mapping for new categories
                    unique_vals = df[col].astype(str).unique()
                    self.label_encoders[col] = {val: i for i, val in enumerate(unique_vals)}
                
                # Apply encoding
                if hasattr(self.label_encoders[col], 'classes_'):
                    # sklearn LabelEncoder
                    df_encoded[col + '_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] if str(x) in self.label_encoders[col].classes_ else -1
                    )
                else:
                    # Simple dict mapping
                    df_encoded[col + '_encoded'] = df[col].astype(str).map(self.label_encoders[col]).fillna(-1)
        
        # Select numerical features
        feature_cols = [
            'operating_hours', 'age_days', 'age_years', 'avg_daily_hours',
            'total_maintenance_count', 'days_since_last_maintenance',
            'avg_maintenance_interval', 'maintenance_frequency',
            'most_common_issue_frequency', 'recent_maintenance_count',
            'issue_diversity', 'type_encoded', 'manufacturer_encoded',
            'model_encoded', 'location_encoded', 'most_common_issue_encoded'
        ]
        
        # Filter to only existing columns
        available_cols = [col for col in feature_cols if col in df_encoded.columns]
        df_numerical = df_encoded[available_cols].copy()
        
        # Handle infinite values and NaN
        df_numerical = df_numerical.replace([np.inf, -np.inf], np.nan)
        df_numerical = df_numerical.fillna(df_numerical.median())
        
        # Ensure all columns are numeric
        for col in df_numerical.columns:
            if not pd.api.types.is_numeric_dtype(df_numerical[col]):
                df_numerical[col] = pd.to_numeric(df_numerical[col], errors='coerce')
        
        df_numerical = df_numerical.fillna(0)
        return df_numerical
    
    def predict_failure(self, features_df, use_ensemble=None):
        """Predict failures using trained models"""
        if self.xgb_time_model is None or self.xgb_type_model is None:
            raise ValueError("Models must be trained first!")
        
        if use_ensemble is None:
            use_ensemble = self.use_ensemble
        
        # Prepare features
        X = self.prepare_features(features_df)
        X_scaled = self.scaler.transform(X)
        
        # XGBoost predictions
        xgb_days_to_failure = self.xgb_time_model.predict(X_scaled)
        xgb_failure_type_encoded = self.xgb_type_model.predict(X_scaled)
        xgb_failure_type_proba = self.xgb_type_model.predict_proba(X_scaled)
        
        # Use ensemble if available
        if use_ensemble and self.rf_time_model is not None:
            rf_days_to_failure = self.rf_time_model.predict(X_scaled)
            final_days_to_failure = 0.7 * xgb_days_to_failure + 0.3 * rf_days_to_failure
            prediction_method = "Ensemble (XGBoost + Random Forest)"
        else:
            final_days_to_failure = xgb_days_to_failure
            prediction_method = "XGBoost"
        
        # Decode failure types
        if hasattr(self.label_encoders.get('failure_type', {}), 'inverse_transform'):
            failure_types = self.label_encoders['failure_type'].inverse_transform(xgb_failure_type_encoded)
        else:
            # Handle dict-based encoding
            failure_type_mapping = self.label_encoders.get('failure_type', {})
            if isinstance(failure_type_mapping, dict):
                reverse_mapping = {v: k for k, v in failure_type_mapping.items()}
                failure_types = [reverse_mapping.get(int(enc), 'Unknown') for enc in xgb_failure_type_encoded]
            else:
                failure_types = ['Unknown'] * len(xgb_failure_type_encoded)
        
        max_proba = np.max(xgb_failure_type_proba, axis=1)
        
        results = []
        for i in range(len(features_df)):
            days = int(max(1, final_days_to_failure[i]))
            result = {
                'equipment_id': features_df.iloc[i]['equipment_id'],
                'days_to_failure': days,
                'predicted_failure_date': pd.Timestamp.now() + pd.Timedelta(days=days),
                'failure_type': failure_types[i],
                'confidence': max_proba[i],
                'risk_level': 'High' if days < 30 else 'Medium' if days < 90 else 'Low',
                'prediction_method': prediction_method
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
# ================================================================================================
# PRODUCTION PREDICTOR CLASS
# ================================================================================================

class ProductionPredictor:
    """Production-ready predictor for equipment failure"""
    
    def __init__(self, model):
        self.model = model
        
    def prepare_features(self, equipment_data):
        """Convert equipment JSON to model features"""
        
        feature_row = {
            'equipment_id': equipment_data.get('id', equipment_data.get('_id', 'unknown')),
            'type': equipment_data.get('type', 'Unknown'),
            'manufacturer': equipment_data.get('manufacturer', 'Unknown'),
            'model': equipment_data.get('modelType', 'Unknown'),
            'location': equipment_data.get('location', 'Unknown'),
            'operating_hours': equipment_data.get('operatingHours', 0),
            'current_status': equipment_data.get('status', 'Unknown')
        }
        
        # Parse installation date
        install_date_str = equipment_data.get('installationDate')
        if install_date_str:
            try:
                install_date = pd.to_datetime(install_date_str)
            except:
                install_date = pd.Timestamp.now() - pd.Timedelta(days=365)
        else:
            install_date = pd.Timestamp.now() - pd.Timedelta(days=365)
        
        # Calculate equipment characteristics
        feature_row['age_days'] = (pd.Timestamp.now() - install_date).days
        feature_row['age_years'] = feature_row['age_days'] / 365.25
        
        if feature_row['age_days'] > 0:
            feature_row['avg_daily_hours'] = feature_row['operating_hours'] / feature_row['age_days']
        else:
            feature_row['avg_daily_hours'] = 0
        
        # Process maintenance history
        maintenance_history = equipment_data.get('maintenanceHistory', [])
        
        if maintenance_history:
            maintenance_df = pd.DataFrame(maintenance_history)
            feature_row['total_maintenance_count'] = len(maintenance_history)
            
            if 'maintenanceDate' in maintenance_df.columns:
                maintenance_dates = pd.to_datetime(maintenance_df['maintenanceDate'])
                last_maintenance = maintenance_dates.max()
                feature_row['days_since_last_maintenance'] = (pd.Timestamp.now() - last_maintenance).days
                
                if len(maintenance_history) > 1:
                    intervals = maintenance_dates.sort_values().diff().dropna().dt.days
                    feature_row['avg_maintenance_interval'] = intervals.mean()
                    feature_row['maintenance_frequency'] = len(maintenance_history) / feature_row['age_days'] * 365
                else:
                    feature_row['avg_maintenance_interval'] = feature_row['age_days']
                    feature_row['maintenance_frequency'] = 1 / feature_row['age_years'] if feature_row['age_years'] > 0 else 0
            else:
                feature_row['days_since_last_maintenance'] = feature_row['age_days']
                feature_row['avg_maintenance_interval'] = feature_row['age_days']
                feature_row['maintenance_frequency'] = 0
            
            if 'issue' in maintenance_df.columns:
                issue_counts = maintenance_df['issue'].value_counts()
                feature_row['most_common_issue'] = issue_counts.index[0]
                feature_row['most_common_issue_frequency'] = issue_counts.iloc[0] / len(maintenance_history)
                feature_row['issue_diversity'] = maintenance_df['issue'].nunique()
            else:
                feature_row['most_common_issue'] = 'General Maintenance'
                feature_row['most_common_issue_frequency'] = 1.0
                feature_row['issue_diversity'] = 1
            
            # Recent maintenance count
            recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            if 'maintenanceDate' in maintenance_df.columns:
                recent_maintenance = maintenance_df[pd.to_datetime(maintenance_df['maintenanceDate']) > recent_cutoff]
                feature_row['recent_maintenance_count'] = len(recent_maintenance)
            else:
                feature_row['recent_maintenance_count'] = 0
        else:
            # No maintenance history
            feature_row['total_maintenance_count'] = 0
            feature_row['days_since_last_maintenance'] = feature_row['age_days']
            feature_row['avg_maintenance_interval'] = feature_row['age_days'] * 2
            feature_row['maintenance_frequency'] = 0
            feature_row['most_common_issue'] = 'None'
            feature_row['most_common_issue_frequency'] = 0
            feature_row['recent_maintenance_count'] = 0
            feature_row['issue_diversity'] = 0
        
        return pd.DataFrame([feature_row])
    
    def predict(self, equipment_data):
        """Make prediction for equipment"""
        try:
            if isinstance(equipment_data, str):
                equipment_data = json.loads(equipment_data)
            
            # Prepare features
            features_df = self.prepare_features(equipment_data)
            
            # Make prediction using trained model
            prediction_df = self.model.predict_failure(features_df, use_ensemble=True)
            
            if len(prediction_df) == 0:
                return {"error": "Prediction failed"}
            
            prediction = prediction_df.iloc[0]
            
            result = {
                "equipment_id": prediction['equipment_id'],
                "prediction": {
                    "days_to_failure": int(prediction['days_to_failure']),
                    "predicted_failure_date": prediction['predicted_failure_date'].strftime('%Y-%m-%d'),
                    "failure_type": prediction['failure_type'],
                    "risk_level": prediction['risk_level'],
                    "confidence": f"{prediction['confidence']:.2%}",
                    "method": prediction['prediction_method']
                },
                "recommendations": self._get_recommendations(prediction)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _get_recommendations(self, prediction):
        """Generate maintenance recommendations"""
        recommendations = []
        risk_level = prediction['risk_level']
        failure_type = prediction['failure_type']
        days = prediction['days_to_failure']
        
        if risk_level == 'High':
            recommendations.append(f"🚨 URGENT: Inspect within {min(7, days)} days")
        elif risk_level == 'Medium':
            recommendations.append(f"⚠️ Schedule maintenance within {min(14, days)} days")
        else:
            recommendations.append("✅ Continue regular maintenance schedule")
        
        # Specific recommendations by failure type
        type_recommendations = {
            'Battery Failure': "🔋 Check battery voltage and capacity",
            'Software Error': "💻 Update firmware to latest version",
            'ECG Lead Detachment': "🔌 Inspect ECG lead connections",
            'Low Suction Pressure': "🔧 Check suction pump and tubing",
            'Occlusion Detected': "🚿 Clean and inspect tubing"
        }
        
        if failure_type in type_recommendations:
            recommendations.append(type_recommendations[failure_type])
        
        return recommendations

# ================================================================================================
# MODEL LOADING
# ================================================================================================

def load_model():
    """Improved model loading with fallbacks"""
    global trained_model, predictor
    
    model_files = [
        'models/equipment_model_fixed.pkl',
        # 'models/equipment_model.pkl', 
        # 'equipment_model_fixed.pkl'
    ]
    
    for model_path in model_files:
        full_path = os.path.join(os.getcwd(), model_path)

        if os.path.exists(full_path):
            try:
                print(f"📥 Loading: {full_path}")
                trained_model = joblib.load(full_path)
                predictor = ProductionPredictor(trained_model)
                print("✅ Model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue
    
    print("❌ No valid model found!")
    return False

# ================================================================================================
# API ROUTES
# ================================================================================================

@app.route('/')
def home():
    """API documentation homepage"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Equipment Failure Prediction API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 25px; }
            .status { padding: 15px; margin: 20px 0; border-radius: 8px; font-weight: bold; }
            .ready { background: #d5edda; color: #155724; border: 1px solid #c3e6cb; }
            .not-ready { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .endpoint { background: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #007bff; }
            .method { background: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
            pre { background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 14px; }
            .url { color: #007bff; font-family: 'Courier New', monospace; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🔮 Equipment Failure Prediction API</h1>
            
            <div class="status {{ status_class }}">
                {{ status_icon }} <strong>Status:</strong> {{ status_message }}
            </div>
            
            <h2>🚀 API Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> <span class="url">/predict</span></h3>
                <p><strong>Predict failure for single equipment</strong></p>
                <p>Send equipment data and get AI-powered failure prediction.</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> <span class="url">/predict-batch</span></h3>
                <p><strong>Batch prediction for multiple equipment</strong></p>
                <p>Process multiple equipment at once for bulk analysis.</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> <span class="url">/health</span></h3>
                <p><strong>API health check</strong></p>
                <p>Check if the API and model are working correctly.</p>
            </div>
            
            <h2>📝 Example Request</h2>
            <pre>curl -X POST {{ request.url_root }}predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "DEF_668288",
    "type": "Defibrillator",
    "manufacturer": "Medtronic",
    "modelType": "LIFEPAK 20e",
    "location": "ICU-Room-1",
    "operatingHours": 23,
    "installationDate": "2025-05-30T00:00:00.000Z",
    "status": "Active",
    "maintenanceHistory": [
      {
        "issue": "Software Error",
        "maintenanceDate": "2025-06-02T00:00:00.000Z"
      }
    ]
  }'</pre>
            
            <h2>📊 Response Format</h2>
            <pre>{
  "equipment_id": "DEF_668288",
  "prediction": {
    "days_to_failure": 45,
    "predicted_failure_date": "2025-07-26",
    "failure_type": "Battery Failure",
    "risk_level": "Medium",
    "confidence": "87.5%",
    "method": "Ensemble (XGBoost + Random Forest)"
  },
  "recommendations": [
    "⚠️ Schedule maintenance within 14 days",
    "🔋 Check battery voltage and capacity"
  ],
  "timestamp": "2025-06-11T10:30:00Z"
}</pre>
        </div>
    </body>
    </html>
    """
    
    if predictor:
        status_class = "ready"
        status_icon = "🟢"
        status_message = "Model loaded and ready for predictions"
    else:
        status_class = "not-ready"
        status_icon = "🔴"
        status_message = "Model not loaded - predictions unavailable"
    
    return render_template_string(html, 
                                status_class=status_class,
                                status_icon=status_icon,
                                status_message=status_message)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200 if predictor else 503

@app.route('/predict', methods=['POST'])
def predict():
    """Single equipment prediction endpoint"""
    if not predictor:
        return jsonify({
            "error": "Model not loaded",
            "message": "Please check server configuration"
        }), 503
    
    try:
        equipment_data = request.get_json()
        
        if not equipment_data:
            return jsonify({
                "error": "No data provided",
                "message": "Please provide equipment data in JSON format"
            }), 400
        
        result = predictor.predict(equipment_data)
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch equipment prediction endpoint"""
    if not predictor:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        equipment_list = data.get('equipment_list', [])
        
        if not equipment_list:
            return jsonify({
                "error": "No equipment list provided",
                "message": "Please provide 'equipment_list' array"
            }), 400
        
        results = []
        for i, equipment in enumerate(equipment_list):
            result = predictor.predict(equipment)
            result['batch_index'] = i
            results.append(result)
        
        return jsonify({
            "results": results,
            "total_processed": len(equipment_list),
            "successful_predictions": len([r for r in results if "error" not in r]),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "message": str(e)
        }), 500

# ================================================================================================
# APPLICATION STARTUP
# ================================================================================================

# Load model when app starts
model_loaded = load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
