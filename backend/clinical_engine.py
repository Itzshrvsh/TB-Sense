import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Constants
CSV_PATH = os.path.join(os.path.dirname(__file__), '../tuberculosis_xray_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/clinical_model.pkl')

# Encoder Mappings
MAPPINGS = {
    'Gender': {'Male': 1, 'Female': 0},
    'Chest_Pain': {'Yes': 1, 'No': 0},
    'Fever': {'None': 0, 'No': 0, 'Mild': 1, 'Moderate': 2, 'High': 3},
    'Night_Sweats': {'Yes': 1, 'No': 0},
    'Sputum_Production': {'None': 0, 'No': 0, 'Low': 1, 'Medium': 2, 'High': 3},
    'Blood_in_Sputum': {'Yes': 1, 'No': 0},
    'Smoking_History': {'Never': 0, 'Former': 1, 'Current': 2},
    'Previous_TB_History': {'Yes': 1, 'No': 0},
    'Class': {'Normal': 0, 'Tuberculosis': 1}
}

class ClinicalRiskEngine:
    def __init__(self):
        self.model = None
        self.max_possible_score = 16
        self.load_model()

    def load_model(self):
        """
        Loads the trained Random Forest classifier.
        """
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("Clinical Random Forest model loaded successfully.")
            except Exception as e:
                print(f"Error loading clinical model: {e}")
        else:
            print("Clinical model file not found. Hybrid fallback active.")

    def train_clinical_model(self, verbose=1):
        """
        Trains a RandomForestClassifier on tuberculosis_xray_dataset.csv and saves to models/
        """
        if not os.path.exists(CSV_PATH):
            print(f"Clinical dataset not found at {CSV_PATH}. Skipping ML training.")
            return False

        print("=" * 60)
        print(" Training Clinical Random Forest Model ")
        print("=" * 60)
        
        try:
            df = pd.read_csv(CSV_PATH)
            
            # Encode categorical features
            encoded_df = df.copy()
            for col, mapping in MAPPINGS.items():
                if col in encoded_df.columns:
                    # Apply mappings, fill unknown values with 0
                    encoded_df[col] = encoded_df[col].map(mapping).fillna(0).astype(int)
            
            feature_cols = [
                'Age', 'Gender', 'Chest_Pain', 'Cough_Severity', 'Breathlessness', 
                'Fatigue', 'Weight_Loss', 'Fever', 'Night_Sweats', 'Sputum_Production', 
                'Blood_in_Sputum', 'Smoking_History', 'Previous_TB_History'
            ]
            
            X = encoded_df[feature_cols]
            y = encoded_df['Class']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Clinical Model Accuracy: {acc:.4f}")
            if verbose:
                print("\nClassification Report:\n", classification_report(y_test, y_pred))
            
            # Ensure models dir exists
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Save
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved clinical model to {MODEL_PATH}")
            self.model = model
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"Error training clinical model: {e}")
            return False

    def calculate_risk(self, patient_data):
        """
        Calculates TB clinical risk using a hybrid Clinical Intelligence Engine:
        60% Medical Knowledge Engine (cumulative evidence points)
        40% Machine Learning Model (Random Forest prediction probability)
        """
        # Form mappings helper
        def get_bool(val):
            if isinstance(val, bool): return val
            if isinstance(val, (int, float)): return bool(val)
            if isinstance(val, str): return val.lower() in ('true', 'yes', 'y', '1', 'on')
            return False

        # Input Extraction
        age = int(patient_data.get('age', 40))
        gender_str = patient_data.get('gender', 'M') # M or F
        gender_val = 'Male' if gender_str == 'M' else 'Female'
        
        chest_pain_bool = get_bool(patient_data.get('chest_pain', False))
        chest_pain = 'Yes' if chest_pain_bool else 'No'
        
        cough_severity = int(patient_data.get('cough_severity', 5))
        cough_duration = int(patient_data.get('cough_duration', 14))
        breathlessness = int(patient_data.get('breathlessness', 3))
        fatigue_severity = int(patient_data.get('fatigue_severity', 5))
        
        # Normalize weight loss
        weight_loss_val = float(patient_data.get('weight_loss_val', 3.0))
        weight_loss_bool = get_bool(patient_data.get('weight_loss', False)) or weight_loss_val > 0.0
        
        fever_level = patient_data.get('fever_level', 'Mild') # Mild, Moderate, High, None
        fever_bool = get_bool(patient_data.get('fever', False)) or fever_level in ('Mild', 'Moderate', 'High')
        
        night_sweats_bool = get_bool(patient_data.get('night_sweats', False))
        night_sweats = 'Yes' if night_sweats_bool else 'No'
        
        sputum_production = patient_data.get('sputum_production', 'Medium') # None, Low, Medium, High
        
        blood_in_sputum_bool = get_bool(patient_data.get('blood_in_sputum', False))
        blood_in_sputum = 'Yes' if blood_in_sputum_bool else 'No'
        
        smoking_history = patient_data.get('smoking_history', 'Never') # Never, Former, Current
        
        previous_tb_bool = get_bool(patient_data.get('previous_tb', False))
        previous_tb = 'Yes' if previous_tb_bool else 'No'
        
        known_lung_disease = get_bool(patient_data.get('known_lung_disease', False))

        # 1. Feature Engineering
        persistent_cough = cough_duration >= 14
        chronic_cough = cough_duration >= 56
        extreme_persistent_cough = cough_duration >= 90
        
        if age <= 18:
            age_group = "0-18"
        elif age <= 35:
            age_group = "19-35"
        elif age <= 60:
            age_group = "36-60"
        else:
            age_group = "60+"
            
        # Count symptoms
        # A symptom is positive if it exceeds a baseline
        has_cough = cough_duration > 0 or cough_severity > 0
        has_breathlessness = breathlessness > 0
        has_fatigue = fatigue_severity > 0 or get_bool(patient_data.get('fatigue', False))
        
        symptoms = [
            has_cough, fever_bool, weight_loss_bool, night_sweats_bool, 
            has_fatigue, blood_in_sputum_bool, chest_pain_bool, has_breathlessness
        ]
        symptom_count = sum(1 for s in symptoms if s)
        
        respiratory_symptoms = [has_cough, has_breathlessness, chest_pain_bool, blood_in_sputum_bool]
        respiratory_symptom_count = sum(1 for s in respiratory_symptoms if s)
        
        tb_warning_signs = [persistent_cough, weight_loss_bool, night_sweats_bool, blood_in_sputum_bool, fever_bool]
        tb_warning_sign_count = sum(1 for s in tb_warning_signs if s)

        # 2. Medical Knowledge Engine (Knowledge-based Scoring)
        points = 0
        contributing_factors = []
        absent_factors = []
        
        # Cough duration points (cumulative)
        cough_points = 0
        if cough_duration >= 90:
            cough_points = 20 + 30 + 40  # cumulative points
            contributing_factors.append(f"Persistent cough >90 days (+40, cumulative: +{cough_points})")
        elif cough_duration >= 56:
            cough_points = 20 + 30
            contributing_factors.append(f"Persistent cough >56 days (+30, cumulative: +{cough_points})")
        elif cough_duration >= 14:
            cough_points = 20
            contributing_factors.append("Persistent cough >14 days (+20)")
        else:
            absent_factors.append("No persistent cough (>14 days) detected")
            
        points += cough_points
        
        # Weight loss
        if weight_loss_bool:
            points += 15
            contributing_factors.append("Weight loss (+15)")
        else:
            absent_factors.append("No weight loss detected")
            
        # Night sweats
        if night_sweats_bool:
            points += 15
            contributing_factors.append("Night sweats (+15)")
        else:
            absent_factors.append("No night sweats detected")
            
        # Blood in sputum
        if blood_in_sputum_bool:
            points += 25
            contributing_factors.append("Blood in sputum (+25)")
        else:
            absent_factors.append("No blood in sputum detected")
            
        # Previous TB history
        if previous_tb_bool:
            points += 20
            contributing_factors.append("Previous TB history (+20)")
        else:
            absent_factors.append("No previous TB history")
            
        # Smoking history
        if smoking_history in ('Former', 'Current'):
            points += 10
            contributing_factors.append(f"Smoking history ({smoking_history}) (+10)")
        else:
            absent_factors.append("No smoking history")
            
        # Fever
        if fever_bool:
            points += 10
            contributing_factors.append(f"Fever ({fever_level}) (+10)")
        else:
            absent_factors.append("No fever detected")
            
        # Fatigue
        if has_fatigue:
            points += 5
            contributing_factors.append("Fatigue (+5)")
        else:
            absent_factors.append("No fatigue detected")

        medical_knowledge_score = min(100.0, points)

        # 3. Machine Learning Model Score
        ml_prob = 0.35  # Default validation fallback
        if self.model:
            try:
                row = pd.DataFrame([{
                    'Age': age,
                    'Gender': MAPPINGS['Gender'][gender_val],
                    'Chest_Pain': MAPPINGS['Chest_Pain'][chest_pain],
                    'Cough_Severity': cough_severity,
                    'Breathlessness': breathlessness,
                    'Fatigue': fatigue_severity,
                    'Weight_Loss': weight_loss_val,
                    'Fever': MAPPINGS['Fever'].get(fever_level, 0),
                    'Night_Sweats': MAPPINGS['Night_Sweats'][night_sweats],
                    'Sputum_Production': MAPPINGS['Sputum_Production'].get(sputum_production, 0),
                    'Blood_in_Sputum': MAPPINGS['Blood_in_Sputum'][blood_in_sputum],
                    'Smoking_History': MAPPINGS['Smoking_History'].get(smoking_history, 0),
                    'Previous_TB_History': MAPPINGS['Previous_TB_History'][previous_tb]
                }])
                ml_prob = float(self.model.predict_proba(row)[0][1])
            except Exception as e:
                print(f"Error executing ML clinical prediction: {e}")

        # 4. Hybrid Calculation (60% Medical Knowledge Engine, 40% Machine Learning Model)
        combined_prob = (0.6 * (medical_knowledge_score / 100.0)) + (0.4 * ml_prob)
        clinical_probability = min(1.0, max(0.0, combined_prob))
        
        # Categorize
        if clinical_probability < 0.35:
            risk_category = "Low Risk"
        elif clinical_probability < 0.65:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
            
        return {
            'clinical_score': int(medical_knowledge_score),
            'risk_category': risk_category,
            'risk_percentage': round(clinical_probability * 100, 2),
            'derived_features': {
                'persistent_cough_flag': persistent_cough,
                'chronic_cough_flag': chronic_cough,
                'extreme_cough_flag': extreme_persistent_cough,
                'symptom_count': symptom_count,
                'respiratory_symptom_count': respiratory_symptom_count,
                'tb_warning_sign_count': tb_warning_sign_count,
                'age_group': age_group,
                'known_lung_disease': known_lung_disease
            },
            'contributing_factors': contributing_factors,
            'absent_factors': absent_factors
        }

if __name__ == '__main__':
    engine = ClinicalRiskEngine()
    # If run directly, execute training as a test
    engine.train_clinical_model()
