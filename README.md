# Kidney Injury Classification

This project implements a deep learning model for kidney injury classification using CT images and corresponding masks in NIfTI format. The model architecture is based on a modified ResNet50 that processes both the CT image and the mask as inputs.

## Project Structure

- `model.py`: Defines the `ResNet50WithMask` model architecture.
- `dataset.py`: Contains the `KidneyDataset` class for loading and preprocessing NIfTI images and masks.
- `train.py`: Implements training and evaluation functions and the training loop.
- `main.py`: Orchestrates data loading, dataset and dataloader creation, and starts the training process.
- `validate.py`: Provides a standalone validation module to evaluate the trained model on the test set.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- nibabel
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install torch torchvision nibabel pandas scikit-learn
```

## Usage

### Training

Run the training pipeline with:

```bash
python main.py
```

Make sure to update the CSV file path and base path for NIfTI files in `main.py` accordingly.

### Validation

After training, evaluate the model on the test set with:

```bash
python validate.py --csv_file path/to/csv --base_path path/to/nifti --model_path best_model.pth
```

## Notes

- The dataset loader handles both 2D and 3D NIfTI images, including those with singleton dimensions.
- The model merges features from the CT image and mask for improved classification.
- The training process saves the best model based on validation accuracy.

