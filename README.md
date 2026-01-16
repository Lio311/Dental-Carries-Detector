# 🦷 Dental Caries Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dental-carries-detector.streamlit.app)

An advanced AI-powered web application for detecting dental caries (cavities) in X-ray images using YOLOv8 deep learning model.
- **Confidence Distribution Charts**: Visual representation of detection quality
- **Detailed Reports**: Exportable JSON reports with complete detection data

### 💾 Export Options
- **Annotated Images**: Download X-rays with highlighted caries areas
- **JSON Reports**: Structured data for integration with other systems

### 🎨 User Experience
- **Professional UI**: Clean, modern interface with custom styling
- **Responsive Design**: Works on desktop and mobile devices
- **Progress Indicators**: Real-time feedback during analysis
- **Error Handling**: Robust validation and user-friendly error messages

---

## 🚀 Quick Start

### Online Demo
Visit our live demo: [Dental Caries Detector](https://dental-carries-detector.streamlit.app)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/dental-caries-detector.git
   cd dental-caries-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   The app will automatically open at `http://localhost:8501`

---

## 📁 Project Structure

```
dental-caries-detector/
├── app.py                          # Main Streamlit application (v2.0)
├── best.pt                         # Trained YOLOv8s model
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (for Streamlit Cloud)
├── README.md                       # This file
└── train_caries.ipynb              # training notebook
```

---

## 🧠 Model Performance

Our **YOLOv8s** model achieves excellent performance on dental X-ray images:

## 🏆 Results Comparison (Optimization Journey)

We have iteratively optimized the training process to maximize performance within 15GB GPU memory.

| Strategy | Model | Image Size | Batch | Memory | mAP50 | Recall | Precision | Notes |
|----------|-------|------------|-------|--------|-------|--------|-----------|-------|
| **Initial** | YOLOv8n | 640px | 64 | ~4GB | 82.6% | 75.7% | 86.2% | First attempt (v1.0) |
| **Baseline** | YOLOv8s | 640px | 16 | ~7GB | **88.3%** | **74.6%** | **97%** | Strong baseline (v2.0) |
| **Balanced** | YOLOv8s | 800px | 10 | ~14.5GB | 87.4% | 77.5% | 93.5% | Better clinical metrics |
| **Strategy C** | YOLOv8m | 640px | 16 | ~14GB | 85.4% | **75.5% | 92.7% | Best Localization (mAP50‑95: 58.0%) |

### Why Strategy C?
While the "Balanced" approach (800px images) improved Precision and Recall significantly, the smaller model (YOLOv8s) struggled with localization (lower mAP50-95).
**Strategy C** switches to a larger model (**YOLOv8m**) with standard 640px images. This gives the model more "brain power" (25M params vs 11M) to understand complex features, which is expected to provide the best overall balance of detection and localization.

### Key Achievements
- 🎯 **97% Precision**: When the model detects caries, it's correct 97% of the time
- 📊 **13.6% improvement** in mAP50-95: Significantly more accurate bounding boxes
- ⚡ **Fast inference**: 4.9ms per image on Tesla T4 GPU
- 🔬 **Clinical reliability**: High precision makes it suitable for medical applications

### Training Details
- **Base Model**: YOLOv8s (Transfer Learning from COCO)
- **Dataset**: 2,706 annotated dental X-ray images
- **Training**: 200 epochs with early stopping (patience=50)
- **Optimizer**: AdamW with learning rate 0.001
- **Augmentations**: 9 types (rotation, scaling, flipping, HSV variations, mosaic, mixup)
- **Training Time**: 2.7 hours on Tesla T4 GPU

---

## 🎯 How to Use

1. **Upload Image(s)**
   - Click the upload button
   - Select one or more dental X-ray images (JPG, JPEG, PNG)
   - Maximum file size: 10MB per image

2. **Adjust Settings**
   - Use the sidebar slider to set confidence threshold
   - Lower threshold = more detections (including uncertain ones)
   - Higher threshold = only high-confidence detections

3. **View Results**
   - Original and annotated images displayed side-by-side
   - Color-coded bounding boxes indicate confidence levels:
     - 🟢 Green: 80-100% (High confidence)
     - 🟡 Yellow: 60-80% (Medium confidence)
     - 🟠 Orange: 40-60% (Low confidence)
     - 🔴 Red: <40% (Very low confidence)

4. **Analyze Statistics**
   - View detection count and confidence metrics
   - Examine confidence distribution chart
   - Review detailed detection list

5. **Download Results**
   - Save annotated images
   - Export JSON reports for documentation

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit 
- **Deep Learning**: Ultralytics YOLOv8s
- **Image Processing**: OpenCV, Pillow
- **Data Visualization**: Plotly
- **Numerical Computing**: NumPy
- **Deployment**: Streamlit Cloud

---

## 📊 Application Features (v2.0)

### Error Handling & Validation
- File size validation (max 10MB)
- Image dimension checks (max 4000px)
- Format verification (JPG, JPEG, PNG)
- Corrupted file detection

### Performance Optimization
- Automatic image resizing for large files
- Efficient memory management
- Cached model loading
- Batch processing support

### User Interface
- Custom CSS styling
- Responsive layout
- Progress bars and spinners
- Informative tooltips
- Professional color scheme
- Statistics dashboard with charts

---

## 🔬 Model Training

The model was trained using advanced techniques optimized for T4 GPU:

### Data Preprocessing
- **Label Fusion**: Merged 12 original classes into single "Caries" class
- **Data Cleaning**: Processed 2,706 annotations across train/val/test sets
- **Quality Control**: Removed corrupted images and invalid labels

### Training Configuration
```python
Model: YOLOv8s
Epochs: 200
Batch Size: 16 (optimized for T4 GPU memory)
Image Size: 640x640
Learning Rate: 0.001 (with warmup)
Optimizer: AdamW
Weight Decay: 0.0005
Patience: 50 (early stopping)
```

### Augmentation Pipeline
```python
degrees=15          # Rotation
translate=0.1       # Translation
scale=0.6           # Scaling
flipud=0.5          # Vertical flip
fliplr=0.5          # Horizontal flip
hsv_h=0.015         # Hue variation
hsv_s=0.7           # Saturation variation
hsv_v=0.4           # Brightness variation
mosaic=1.0          # Mosaic augmentation
mixup=0.05          # Mixup augmentation
```

### Improvements Over Baseline
- ✅ Upgraded from YOLOv8n to YOLOv8s (3.7x more parameters)
- ✅ Enhanced augmentation pipeline (9 augmentations)
- ✅ Optimized hyperparameters for stability
- ✅ Extended training duration (120 → 200 epochs)
- ✅ Memory-optimized for T4 GPU constraints

For detailed training information, see:
- `train_caries.ipynb` - Latest optimized training

---

## 🚢 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Update application"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

3. **Required Files**
   - `app.py` - Main application
   - `best.pt` - Trained model (must be in repo)
   - `requirements.txt` - Python dependencies
   - `packages.txt` - System dependencies

### Environment Requirements
- Python 3.10+
- CUDA support (optional, for GPU acceleration)
- 2GB+ RAM recommended

---

## 📝 Usage Examples

### Single Image Analysis
```python
# Upload one X-ray image
# Adjust confidence threshold to 25%
# View results and statistics
# Download annotated image
```

### Integration with Other Systems
```python
# Export JSON reports
# Parse detection data
# Integrate with dental management software
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **Dataset**: Roboflow Universe - Dental Caries X-ray Dataset
- **Model**: Ultralytics YOLOv8
- **Framework**: Streamlit
- **Training Platform**: Google Colab (Tesla T4 GPU)

---

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

## 🔄 Version History

### v2.2 (Strategy C) - November 2025
- 🚀 **Strategic Upgrade to YOLOv8m**
- 🧠 Capacity-optimized training (640px images + 25M parameters)
- 🎯 Target: >90% mAP50 with high clinical precision
- 🔄 Current active strategy

### v2.1 (Balanced Optimization) - November 2025
- ⚖️ **Balanced Config**: YOLOv8s with 800px images
- 🏥 **Clinical Improvements**: 93.5% Precision, 77.5% Recall
- 📉 Slight mAP decrease (87.4%) due to localization challenges
- 💾 Memory optimized (Batch 10) to fit 15GB
- 
### v2.0 (Current) - November 2025
- 🎯 **Upgraded to YOLOv8s model** (+13.6% mAP50-95, +10.8% Precision)
- ✨ Complete UI redesign with professional styling
- 📊 Added statistics dashboard and charts
- 💾 Implemented download functionality for images and reports
- 🔄 Added batch processing support
- ⚡ Performance optimizations and error handling
- 📱 Improved mobile responsiveness
- 🧠 Extended training to 200 epochs with optimized hyperparameters

### v1.0 - October 2025
- 🎯 Initial release with YOLOv8n model
- 🎨 Simple color-coded bounding boxes
- ⚙️ Confidence threshold slider
- 📊 Basic detection metrics (82.6% mAP50, 44.2% mAP50-95)

---

<div align="center">

**Made with ❤️ using Streamlit and YOLOv8**

**Current Model: YOLOv8s | mAP50: 88.3% | Precision: 97.0%**

</div>











