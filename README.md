# Federated Learning for Bladder Cancer Detection

![Demo GIF](./animation.gif)

This is quite an old and so-to-say a failed project due to a failed professor and his failed methods (No Names Taken, iykyk!!), however, a lot of things to learn here.

Well, contrary to the animation, I have 4 fictional centers in this repository. It contains implementations of federated learning approaches for multi-center bladder cancer image classification. The project focuses on training distributed models across multiple medical centers while preserving data privacy.

## Project Structure

```
├── Innovative_Assignment/
│   ├── main.py
│   ├── requirements.txt
│   ├── Center_1/
│   ├── Center_2/
│   ├── Center_3/
│   └── Center_4/
│       └── [Diverticulosis, Neoplasm, Peritonitis, Ureters folders]
├── Practicals/
│   ├── main.py
│   ├── Prac1_22BCE335_AI3.ipynb through Prac10_22BCE335_AI3.ipynb
└── ppt/
```

## Dataset

The project uses a **Bladder Cancer Detection** dataset organized across 4 medical centers. Each center contains images categorized into 4 classes:
- **Diverticulosis** (Label: 0)
- **Neoplasm** (Label: 1)
- **Peritonitis** (Label: 2)
- **Ureters** (Label: 3)

**Dataset Link**: [Kaggle - Bladder Cancer Classification](https://www.kaggle.com/datasets/shirtgm/bladder-cancer-classification)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r Innovative_Assignment/requirements.txt
```

## Key Dependencies

- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `opencv-python` - Image processing
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning metrics
- `SimpleITK` - Medical image processing
- `scipy` - Scientific computing

## Usage

### Running the Main Assignment

```bash
cd Innovative_Assignment
python main.py
```

### Running Practical Notebooks

Open the Jupyter notebooks in the `Practicals/` folder:

```bash
jupyter notebook Practicals/Prac1_22BCE335_AI3.ipynb
```

## Main Components

### Custom Dataset Class
Loads images from multi-center folders and assigns class labels:
```python
class custom_dataset():
    def __init__(self, folder_path=None, transform=None)
```

### CNN Model Architecture
- 3 convolutional layers with batch normalization
- Max pooling and adaptive average pooling
- 2 fully connected layers (128 → 512 → 4)
- Dropout (0.5) for regularization

### Federated Learning Framework
- Distributed training across 4 centers
- Global model aggregation
- Local model training with center-specific data
- Multiple rounds of federated learning

## Features

✅ **Multi-center Data Handling** - Support for distributed datasets across multiple medical centers

✅ **Federated Learning** - Train models while preserving data privacy

✅ **Data Augmentation** - Random rotations, flips, and color jitter for improved generalization

✅ **Performance Metrics** - Accuracy, Precision, Recall, and F1-Score tracking

✅ **Batch Normalization** - Stabilized training with custom weight initialization

✅ **Model Evaluation** - Comprehensive validation and testing protocols

## Model Training

The training process includes:
1. Local training on each center's data
2. Global model aggregation across centers
3. Multiple federated learning rounds
4. Evaluation on test set

### Training Parameters
- **Batch Size**: 100
- **Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 2 per round
- **Federated Rounds**: 5-10
- **Image Size**: 320×320

## Results & Metrics

The model is evaluated using:
- **Accuracy** - Overall correctness
- **Precision** - True positive rate among predicted positives
- **Recall** - True positive rate among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Loss** - Cross-entropy loss

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| Prac1 | Dataset loading and exploration from multiple centers |
| Prac2 | Federated learning framework implementation |
| Prac3 | Data preprocessing and EDA |
| Prac4 | Model architecture with custom initialization |
| Prac5 | Data augmentation and transformation pipelines |
| Prac6 | Advanced federated learning with weight initialization |
| Prac7 | Training with metrics tracking and visualization |
| Prac8 | Federated learning with model compression |
| Prac9 | Non-IID data handling and privacy-preserving techniques |
| Prac10 | Pre-trained models and transfer learning |

## Technical Details

### Image Preprocessing
- Resize to 320×320
- Random horizontal/vertical flips
- Random rotation (±15 degrees)
- Color jitter (brightness & contrast)
- Normalization with standard ImageNet parameters

### Weight Initialization
- **Conv Layers**: Xavier uniform initialization
- **Batch Norm**: Weight=1, Bias=0.2
- **Linear Layers**: Xavier normal initialization, Bias=0.3

## Notes

- The project is designed for Google Colab environments (references to `/content/drive/MyDrive/`)
- Modify paths according to your local directory structure
- GPU acceleration recommended for faster training