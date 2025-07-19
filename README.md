# Cat vs Dog Image Classifier 🐱🐶

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of cats and dogs with high accuracy.

## 📋 Project Overview

This project implements a binary image classification model that can distinguish between cats and dogs. The CNN architecture includes multiple convolutional layers, pooling layers, and dense layers to effectively learn features from pet images.

## 🏗️ Model Architecture

```
Sequential Model:
├── Conv2D (32 filters, 3x3 kernel, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (32 filters, 3x3 kernel, ReLU) 
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (128 units, ReLU)
└── Dense (1 unit, Sigmoid)
```

## 📊 Dataset Information

- **Training Set**: 8,000 images (4,000 cats + 4,000 dogs)
- **Test Set**: 2,000 images (1,000 cats + 1,000 dogs)
- **Image Size**: 64x64 pixels
- **Format**: RGB images

## 🚀 Features

- **Data Augmentation**: Rescaling, shearing, zooming, and horizontal flipping
- **Binary Classification**: Outputs probability for cat (0) or dog (1)
- **Real-time Prediction**: Can classify new images instantly
- **Optimized Architecture**: Efficient CNN design for quick training

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
NumPy
Keras
```

### Install Dependencies
```bash
pip install tensorflow numpy keras pillow
```

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/cat-dog-classifier.git
cd cat-dog-classifier
```

## 📁 Project Structure

```
cat-dog-classifier/
│
├── cnn.ipynb                 # Main Jupyter notebook
├── dataset/
│   ├── training_set/
│   │   ├── cats/            # 4,000 cat images
│   │   └── dogs/            # 4,000 dog images
│   ├── test_set/
│   │   ├── cats/            # 1,000 cat images
│   │   └── dogs/            # 1,000 dog images
│   └── single_prediction/
│       ├── cat_or_dog_1.jpg # Sample cat image
│       └── cat_or_dog_2.jpg # Sample dog image
├── models/
│   └── saved_model/         # Trained model files
├── README.md
└── requirements.txt
```

## 🎯 Usage

### Training the Model
```python
# Run the Jupyter notebook
jupyter notebook cnn.ipynb

# Or run as Python script
python cnn.py
```

### Making Predictions
```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('models/cat_dog_classifier.h5')

# Load and preprocess image
test_image = image.load_img('path/to/image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make prediction
result = model.predict(test_image)
prediction = 'dog' if result[0][0] > 0.5 else 'cat'
print(f"Prediction: {prediction}")
```

## 📈 Model Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- **Epochs**: 25
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|--------|
| Batch Size | 32 |
| Image Size | 64x64 |
| Epochs | 25 |
| Learning Rate | Default (Adam) |
| Filters (Conv1) | 32 |
| Filters (Conv2) | 32 |
| Dense Units | 128 |

## 🎨 Data Augmentation

The model uses several augmentation techniques to improve generalization:
- **Rescaling**: Normalizes pixel values (1./255)
- **Shear Range**: 0.2
- **Zoom Range**: 0.2  
- **Horizontal Flip**: Random horizontal flipping

## 🚧 Future Improvements

- [ ] Implement transfer learning with pre-trained models (VGG16, ResNet)
- [ ] Add more animal classes (birds, rabbits, etc.)
- [ ] Increase image resolution for better accuracy
- [ ] Deploy model as web application
- [ ] Add model visualization and interpretability features

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [Ahmadhammam03](https://github.com/Ahmadhammam03)
- LinkedIn: [Ahmad Hammam](www.linkedin.com/in/ahmad-hammam-1561212b2)

## 🙏 Acknowledgments

- TensorFlow/Keras documentation
- Deep Learning community
- Dataset providers

---

⭐ **If you found this project helpful, please give it a star!** ⭐
