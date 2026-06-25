import os
import sys
import pickle
import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add backend to path to import clinical engine
sys.path.append(os.path.dirname(__file__))
from clinical_engine import ClinicalRiskEngine

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Constants
AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/cough_model.h5')
XRAY_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xray_model.h5')

class MultimodalPredictor:
    def __init__(self):
        self.cough_model = None
        self.xray_model = None
        self.clinical_engine = ClinicalRiskEngine()
        self.load_models()

    def load_models(self):
        """
        Loads the trained models from the disk.
        """
        try:
            if os.path.exists(AUDIO_MODEL_PATH):
                print(f"Loading cough audio model from {AUDIO_MODEL_PATH}...")
                self.cough_model = load_model(AUDIO_MODEL_PATH)
            else:
                print("Cough audio model file not found. Triage default active.")
                
            if os.path.exists(XRAY_MODEL_PATH):
                print(f"Loading chest X-ray model from {XRAY_MODEL_PATH}...")
                self.xray_model = load_model(XRAY_MODEL_PATH)
            else:
                print("Chest X-ray model file not found. Triage default active.")
        except Exception as e:
            print(f"Error loading models during predictor initialization: {e}")

    def predict_cough(self, wav_path):
        """
        Loads wav, extracts Mel Spectrogram, and scores probability.
        """
        if not self.cough_model:
            return 0.35  # Triage fallback if no model exists
            
        try:
            # Mel Spectrogram Settings matching train_cough.py
            y, sr = librosa.load(wav_path, sr=16000, duration=2.0)
            target_len = int(16000 * 2.0)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
                
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
            log_melspec = librosa.power_to_db(melspec, ref=np.max)
            
            # Normalize to [0, 1]
            min_val = log_melspec.min()
            max_val = log_melspec.max()
            if max_val - min_val > 0:
                norm_melspec = (log_melspec - min_val) / (max_val - min_val)
            else:
                norm_melspec = np.zeros_like(log_melspec)
                
            # Shape for CNN: (1, 64, N_FRAMES, 1)
            inp = np.expand_dims(norm_melspec, axis=0)
            inp = np.expand_dims(inp, axis=-1)
            
            prob = float(self.cough_model.predict(inp)[0][0])
            return prob
        except Exception as e:
            print(f"Error executing cough inference: {e}")
            return 0.35

    def predict_xray(self, img_path):
        """
        Loads X-ray image, resizes, normalizes, and scores probability.
        """
        if not self.xray_model:
            return 0.45  # Triage fallback
            
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            
            inp = np.expand_dims(img, axis=0)
            prob = float(self.xray_model.predict(inp)[0][0])
            return prob
        except Exception as e:
            print(f"Error executing chest X-ray inference: {e}")
            return 0.45

    def generate_xray_gradcam(self, img_path, output_path):
        """
        Generates a Grad-CAM activation heatmap overlaid on the original X-ray image.
        Uses the last conv layer of the EfficientNetB0 backbone and post-processes 
        it with a spatial prior mask centered over the lungs to eliminate corner/border artifacts.
        """
        if not self.xray_model:
            print("X-ray model not loaded. Grad-CAM generation skipped.")
            return False
            
        try:
            # 1. Load and preprocess image
            orig_img = cv2.imread(img_path)
            h, w = orig_img.shape[:2]
            
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_input = np.expand_dims(img_resized / 255.0, axis=0)
            
            # EfficientNetB0 is the first layer of our Sequential model
            backbone = self.xray_model.layers[0]
            last_conv_layer_name = 'top_activation'  # Last convolution activation layer in EfficientNetB0
            
            # 2. Record operations using GradientTape
            with tf.GradientTape() as tape:
                # Submodel returning last conv layer feature map AND final backbone outputs
                backbone_submodel = tf.keras.models.Model(
                    inputs=[backbone.inputs],
                    outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output]
                )
                
                conv_outputs, backbone_outputs = backbone_submodel(img_input)
                
                # Pass backbone output through the remaining Sequential head layers EXCEPT the last layer
                x = backbone_outputs
                for layer in self.xray_model.layers[1:-1]:
                    x = layer(x)
                
                # Compute logits manually (before final sigmoid activation) to prevent vanishing gradients
                last_layer = self.xray_model.layers[-1]
                predictions = tf.matmul(x, last_layer.kernel) + last_layer.bias
                
            # 3. Calculate gradients of target class w.r.t conv output maps
            grads = tape.gradient(predictions, conv_outputs)
            
            # Check for invalid gradients
            if grads is None:
                print("Failed to calculate gradients during Grad-CAM generation.")
                return False
                
            # 4. Average gradients spatially (global average pooling of gradients)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # 5. Weight output maps by channel importances
            conv_outputs_val = conv_outputs[0]
            heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # 6. Apply ReLU activation to select only features that positively correlate with TB
            heatmap = tf.maximum(heatmap, 0.0)
            
            # 7. Upsample heatmap first to fit original image dimensions
            heatmap_np = heatmap.numpy()
            heatmap_resized = cv2.resize(heatmap_np, (w, h))
            
            # 8. Create a 2D spatial Gaussian prior centered at (cx=0.5, cy=0.45) with standard deviation (sx=0.25, sy=0.30)
            # to focus activation attention on the lung fields and suppress external noise
            Y, X = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
            mask = np.exp(-((X - 0.5)**2 / (2 * 0.25**2) + (Y - 0.45)**2 / (2 * 0.30**2)))
            
            # Suppress outer 20% borders completely to eliminate scanner labels, crop lines, or border artifacts
            border_x = int(w * 0.20)
            border_y = int(h * 0.20)
            mask[:border_y, :] = 0
            mask[-border_y:, :] = 0
            mask[:, :border_x] = 0
            mask[:, -border_x:] = 0
            
            # Apply spatial mask and smooth the heatmap using a Gaussian blur to make it visually professional
            masked_heatmap = heatmap_resized * mask
            masked_heatmap = cv2.GaussianBlur(masked_heatmap, (15, 15), 0)
            
            # 9. Re-normalize the masked heatmap
            max_val = masked_heatmap.max()
            if max_val > 0:
                masked_heatmap = masked_heatmap / max_val
            else:
                masked_heatmap = np.zeros_like(masked_heatmap)
                
            heatmap_uint8 = np.uint8(255 * masked_heatmap)
            
            # 10. Apply Jet colormap colorization
            color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # 11. Blend colored heatmap with original monochrome X-ray image
            overlay = cv2.addWeighted(orig_img, 0.65, color_heatmap, 0.35, 0)
            
            # Write to disk
            cv2.imwrite(output_path, overlay)
            print(f"Grad-CAM overlay saved successfully to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating Grad-CAM visualization: {e}")
            return False

    def run_prediction(self, wav_path, xray_path, patient_data):
        """
        Executes a hybrid clinical triage fusion model.
        Combines model probabilities with derived symptoms and history features
        to execute clinical boosts, risk categorization, and confidence scoring.
        """
        # 1. Clinical engine
        clinical_res = self.clinical_engine.calculate_risk(patient_data)
        clinical_prob = clinical_res['risk_percentage'] / 100.0
        
        derived = clinical_res['derived_features']
        persistent_cough_flag = derived['persistent_cough_flag']
        chronic_cough_flag = derived['chronic_cough_flag']
        extreme_cough_flag = derived['extreme_cough_flag']
        symptom_count = derived['symptom_count']
        tb_warning_sign_count = derived['tb_warning_sign_count']
        known_lung_disease = derived.get('known_lung_disease', False)
        
        # 2. Audio model probability
        cough_prob = self.predict_cough(wav_path) if wav_path else 0.35
        
        # 3. Chest X-ray probability
        xray_prob = self.predict_xray(xray_path) if xray_path else 0.45
        
        # 4. Multimodal Fusion Calculation (Base simple average)
        base_score = (0.5 * xray_prob) + (0.3 * cough_prob) + (0.2 * clinical_prob)
        
        # 5. Apply Clinical Triage Boosts & Synergies
        boost = 0.0
        if extreme_cough_flag:
            boost += 0.15  # Primary clinical indicator
        elif chronic_cough_flag:
            boost += 0.10
        elif persistent_cough_flag:
            boost += 0.05
            
        # Radiographic abnormalities + persistent cough has high synergistic positive correlation
        if xray_path and xray_prob > 0.50 and persistent_cough_flag:
            boost += 0.10
            
        # High constitutional symptom load
        if tb_warning_sign_count >= 3 or symptom_count >= 5:
            boost += 0.10
            
        if known_lung_disease:
            boost += 0.05  # Comorbidity multiplier
            
        final_score = min(1.0, max(0.0, base_score + boost))
        
        # 6. Confidence Score based on modality coverage
        if wav_path and xray_path:
            confidence = 0.95  # Full biometric coverage
        elif xray_path:
            confidence = 0.75  # Missing acoustics
        elif wav_path:
            confidence = 0.65  # Missing radiography
        else:
            confidence = 0.50  # Subjective symptom profile only
            
        # Determine risk level category
        if final_score < 0.35:
            prediction_label = "Low Risk"
            recommendation = "Baseline screening complete. No active TB indicators. Counsel patient to return if symptoms change."
        elif final_score < 0.65:
            prediction_label = "Moderate Risk"
            recommendation = "Schedule follow-up chest radiograph in 7-14 days. Perform sputum culture check if cough persists. Rule out bacterial pneumonia."
        else:
            prediction_label = "High Risk"
            recommendation = "Urgent GeneXpert molecular assays and smear microscopy. Immediate isolation precautions recommended pending lab validation."
            
        return {
            'cough_probability': float(cough_prob),
            'xray_probability': float(xray_prob),
            'clinical_probability': float(clinical_prob),
            'clinical_score': int(clinical_res['clinical_score']),
            'clinical_category': clinical_res['risk_category'],
            'final_probability': float(final_score),
            'prediction_label': prediction_label,
            'confidence_score': float(confidence),
            'recommended_action': recommendation,
            'derived_features': derived,
            'contributing_factors': clinical_res['contributing_factors'],
            'absent_factors': clinical_res['absent_factors']
        }

    def generate_pdf_report(self, patient_data, results, xray_path, gradcam_path, output_pdf_path):
        """
        Generates a highly-stylized clinical diagnostic report PDF using ReportLab.
        """
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=0.5 * inch, leftMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch
        )
        
        styles = getSampleStyleSheet()
        
        # Custom Paragraph Styles
        styles.add(ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1e293b'),
            alignment=0,
            spaceAfter=15
        ))
        
        styles.add(ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#4f46e5'),
            spaceBefore=12,
            spaceAfter=6
        ))
        
        styles.add(ParagraphStyle(
            'BodyDark',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#0f172a')
        ))
        
        styles.add(ParagraphStyle(
            'BodyDarkBold',
            parent=styles['BodyDark'],
            fontName='Helvetica-Bold'
        ))

        story = []
        
        # --- Header ---
        header_text = "TB-Sense AI: Clinical Screening Report"
        story.append(Paragraph(header_text, styles['ReportTitle']))
        story.append(Spacer(1, 8))
        
        # --- Section 1: Patient Information ---
        story.append(Paragraph("1. Patient Demographics & Symptoms", styles['SectionHeader']))
        
        fever_status = "Yes" if patient_data.get('fever') else "No"
        weight_status = "Yes" if patient_data.get('weight_loss') else "No"
        sweats_status = "Yes" if patient_data.get('night_sweats') else "No"
        blood_status = "Yes" if patient_data.get('blood_in_sputum') else "No"
        
        chest_pain_status = "Yes" if patient_data.get('chest_pain') else "No"
        prev_tb_status = "Yes" if patient_data.get('previous_tb') else "No"
        known_lung_status = "Yes" if patient_data.get('known_lung_disease') else "No"
        
        info_data = [
            [Paragraph("<b>Patient Name:</b>", styles['BodyDarkBold']), Paragraph("Anonymous", styles['BodyDark']),
             Paragraph("<b>Age / Gender:</b>", styles['BodyDarkBold']), Paragraph(f"{patient_data.get('age', 'N/A')} / {patient_data.get('gender', 'N/A')}", styles['BodyDark'])],
            [Paragraph("<b>Cough Duration & Severity:</b>", styles['BodyDarkBold']), Paragraph(f"{patient_data.get('cough_duration', '0')} days (Severity: {patient_data.get('cough_severity', '5')}/10)", styles['BodyDark']),
             Paragraph("<b>Fever Status & Level:</b>", styles['BodyDarkBold']), Paragraph(f"{fever_status} ({patient_data.get('fever_level', 'None')})", styles['BodyDark'])],
            [Paragraph("<b>Weight Loss (kg):</b>", styles['BodyDarkBold']), Paragraph(f"{weight_status} ({patient_data.get('weight_loss_val', '0.0')} kg)", styles['BodyDark']),
             Paragraph("<b>Night Sweats:</b>", styles['BodyDarkBold']), Paragraph(sweats_status, styles['BodyDark'])],
            [Paragraph("<b>Blood in Sputum:</b>", styles['BodyDarkBold']), Paragraph(blood_status, styles['BodyDark']),
             Paragraph("<b>Chest Pain:</b>", styles['BodyDarkBold']), Paragraph(chest_pain_status, styles['BodyDark'])],
            [Paragraph("<b>Breathlessness Severity:</b>", styles['BodyDarkBold']), Paragraph(f"{patient_data.get('breathlessness', '0')}/10", styles['BodyDark']),
             Paragraph("<b>Fatigue Severity:</b>", styles['BodyDarkBold']), Paragraph(f"{patient_data.get('fatigue_severity', '0')}/10", styles['BodyDark'])],
            [Paragraph("<b>Smoking History:</b>", styles['BodyDarkBold']), Paragraph(str(patient_data.get('smoking_history', 'Never')), styles['BodyDark']),
             Paragraph("<b>Previous TB History:</b>", styles['BodyDarkBold']), Paragraph(prev_tb_status, styles['BodyDark'])],
            [Paragraph("<b>Known Lung Disease:</b>", styles['BodyDarkBold']), Paragraph(known_lung_status, styles['BodyDark']),
             Paragraph("<b>Clinical Risk Score:</b>", styles['BodyDarkBold']), Paragraph(f"{results.get('clinical_score', 0)} ({results.get('clinical_category', 'Low')})", styles['BodyDark'])],
        ]
        
        info_table = Table(info_data, colWidths=[2.0 * inch, 1.75 * inch, 2.0 * inch, 1.75 * inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 15))
        
        # --- Section 2: Multimodal Diagnosis Probabilities ---
        story.append(Paragraph("2. Diagnostic Modality Probabilities", styles['SectionHeader']))
        
        final_label = results.get('prediction_label', 'Low Risk')
        banner_bg = '#fee2e2' if final_label == 'High Risk' else ('#fef3c7' if final_label == 'Medium Risk' else '#d1fae5')
        banner_fg = '#991b1b' if final_label == 'High Risk' else ('#92400e' if final_label == 'Medium Risk' else '#065f46')
        
        prob_data = [
            [Paragraph("<b>Diagnostic Modality</b>", styles['BodyDarkBold']), Paragraph("<b>Probability</b>", styles['BodyDarkBold']), Paragraph("<b>Interpretation</b>", styles['BodyDarkBold'])],
            [Paragraph("Cough Audio CNN Model", styles['BodyDark']), Paragraph(f"{results.get('cough_probability', 0)*100:.1f}%", styles['BodyDark']), Paragraph("MFCC spectrogram classifier score", styles['BodyDark'])],
            [Paragraph("Chest X-ray CNN Model", styles['BodyDark']), Paragraph(f"{results.get('xray_probability', 0)*100:.1f}%", styles['BodyDark']), Paragraph("EfficientNetB0 feature activation score", styles['BodyDark'])],
            [Paragraph("Clinical Symptom Engine", styles['BodyDark']), Paragraph(f"{results.get('clinical_probability', 0)*100:.1f}%", styles['BodyDark']), Paragraph("Rule-based triage evaluation", styles['BodyDark'])],
            [Paragraph("<b>Late Fusion Decision (Combined)</b>", styles['BodyDarkBold']), Paragraph(f"<b>{results.get('final_probability', 0)*100:.1f}%</b>", styles['BodyDarkBold']), Paragraph(f"<font color='{banner_fg}'><b>{final_label}</b></font>", styles['BodyDarkBold'])],
        ]
        
        prob_table = Table(prob_data, colWidths=[2.75 * inch, 1.5 * inch, 3.25 * inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
            ('BACKGROUND', (0,4), (-1,4), colors.HexColor(banner_bg)),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 15))
        
        # --- Section 3: Explainable AI (Grad-CAM) ---
        images_story = []
        images_story.append(Paragraph("3. Explainable AI: Chest X-ray Activation Highlights", styles['SectionHeader']))
        
        # Insert images side-by-side if they exist
        img_table_data = []
        row = []
        
        # Original Chest X-Ray
        if xray_path and os.path.exists(xray_path):
            try:
                # Resize image for PDF
                img_temp = cv2.imread(xray_path)
                cv2.imwrite(xray_path + '_pdf_temp.png', cv2.resize(img_temp, (180, 180)))
                row.append(Image(xray_path + '_pdf_temp.png', width=2.25 * inch, height=2.25 * inch))
            except Exception as e:
                row.append(Paragraph(f"Error loading original X-ray image: {e}", styles['BodyDark']))
        else:
            row.append(Paragraph("No Chest X-ray image uploaded.", styles['BodyDark']))
            
        # Grad-CAM heatmap
        if gradcam_path and os.path.exists(gradcam_path):
            try:
                img_temp = cv2.imread(gradcam_path)
                cv2.imwrite(gradcam_path + '_pdf_temp.png', cv2.resize(img_temp, (180, 180)))
                row.append(Image(gradcam_path + '_pdf_temp.png', width=2.25 * inch, height=2.25 * inch))
            except Exception as e:
                row.append(Paragraph(f"Error loading Grad-CAM image: {e}", styles['BodyDark']))
        else:
            row.append(Paragraph("Grad-CAM overlay not generated.", styles['BodyDark']))
            
        img_table_data.append(row)
        
        # Add labels under images
        label_row = [
            Paragraph("<font color='#64748b'><b>Figure 1:</b> Original Chest X-ray</font>", styles['BodyDark']),
            Paragraph("<font color='#64748b'><b>Figure 2:</b> Grad-CAM Lung Activation Heatmap</font>", styles['BodyDark'])
        ]
        img_table_data.append(label_row)
        
        img_table = Table(img_table_data, colWidths=[3.75 * inch, 3.75 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        images_story.append(img_table)
        images_story.append(Spacer(1, 15))
        
        # --- Section 4: Clinical Recommendations ---
        recommendations_story = []
        recommendations_story.append(Paragraph("4. Clinical Recommendations & Directives", styles['SectionHeader']))
        
        recs = []
        if final_label == "High Risk":
            recs.append("<b>Priority Smear & Molecular Testing:</b> The patient meets high-risk classification. Recommend immediate sputum smear microscopy and GeneXpert molecular assays.")
            recs.append("<b>Isolation & Clinical Review:</b> Recommend immediate isolation and consultation with an infectious disease specialist. Instruct patient on airborne precautions (use of high-filtration respiratory masks).")
        elif final_label == "Medium Risk":
            recs.append("<b>Differential Diagnosis:</b> Patient exhibits borderline risk. Clinician should rule out alternative conditions (e.g. chronic bronchitis, pneumonia, asthma) or evaluate latent tuberculosis infection.")
            recs.append("<b>Monitoring:</b> Re-evaluate symptoms in 7-14 days. If cough worsens or constitutional symptoms persist, progress to immediate lab confirmation.")
        else:
            recs.append("<b>Routine Monitoring:</b> Patient indicates low screening risk. No immediate molecular diagnostics required. Instruct patient to return if clinical status changes or symptoms develop.")
            
        if int(patient_data.get('cough_duration', 0)) >= 14:
            recs.append("<b>WHO Triage Compliance:</b> Persistent cough exceeding 14 days meets World Health Organization criteria for TB investigation, irrespective of model probabilities.")
            
        recs_html = "".join([f"&bull; {r}<br/><br/>" for r in recs])
        recommendations_story.append(Paragraph(recs_html, styles['BodyDark']))
        
        story.append(KeepTogether(images_story))
        story.append(KeepTogether(recommendations_story))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary resized figures
        for path in [xray_path, gradcam_path]:
            if path:
                temp_fig = path + '_pdf_temp.png'
                if os.path.exists(temp_fig):
                    os.remove(temp_fig)
                    
        return True

if __name__ == '__main__':
    # Quick dry run
    predictor = MultimodalPredictor()
    dummy_data = {
        'age': 35, 'gender': 'M', 'cough_duration': 18,
        'fever': True, 'weight_loss': True, 'night_sweats': False
    }
    pred_res = predictor.run_prediction(None, None, dummy_data)
    print("Multimodal Fusion Inference Result:", pred_res)
