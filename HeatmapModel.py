import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from fastai.vision.all import *
from fastai.metrics import error_rate
from pathlib import Path
import cv2
from scipy import ndimage
from scipy.spatial.distance import euclidean
from skimage import measure, morphology
from sklearn.decomposition import PCA
import math
import torch

BASE_DIR = Path(__file__).resolve().parent
PATH = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_NAME = "corneal_classifier.pkl"

class CornealAnalyzer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.learn = None
        self._load_or_create_model()

    def _ensure_models_directory(self):
        MODELS_DIR.mkdir(exist_ok=True)

    def _load_or_create_model(self):
        self._ensure_models_directory()
        model_file = MODELS_DIR / MODEL_NAME

        print(f"Looking for model at: {model_file.resolve()}")
        if model_file.exists():
            print("Preexisting model is found")
            self.learn = load_learner(model_file)
        else:
            self._create_and_train_model()

    def _create_data_loaders(self, batch_size=16, image_size=224):
        if not PATH.exists():
            raise FileNotFoundError(f"Dataset directory {PATH} not found!")

        astigmatic_dir = PATH / "Astigmatism"
        normal_dir = PATH / "Healthy"

        if not astigmatic_dir.exists() or not normal_dir.exists():
            raise FileNotFoundError("Required subdirectories 'Astigmatism' and 'Healthy' not found!")

        data = ImageDataLoaders.from_folder(
            PATH,
            train=".",
            valid_pct=0.2,
            seed=42,
            item_tfms=Resize(image_size),
            batch_tfms=aug_transforms(size=image_size, mult=0.5),
            bs=batch_size,
            num_workers=0,
            pin_memory=False
        )

        return data

    def _create_and_train_model(self):
        print("Creating data loaders...")
        data = self._create_data_loaders()

        # Use CPU if CUDA causes issues
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        self.learn = vision_learner(data, resnet34, metrics=accuracy)

        # Move to appropriate device
        if device == 'cpu':
            self.learn.model = self.learn.model.cpu()

        # Find optimal learning rate
        try:
            lr_suggestion = self.learn.lr_find()
        except Exception as e:
            print(f"Learning rate finder failed: {e}")
            print("Using default learning rate...")

        # Train the model with error handling
        print("Training model...")
        try:
            self.learn.fit_one_cycle(8, lr_max=slice(3e-4, 3e-3))
        except Exception as e:
            print(f"Training with fit_one_cycle failed: {e}")
            print("Trying alternative training method...")
            self.learn.fine_tune(5, base_lr=3e-4)

        # Save the model
        model_path = MODELS_DIR / MODEL_NAME
        self.learn.export(model_path)
        print(f"Model saved to {Path(__file__) / model_path}")
    def _preprocess_heightmap(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        img_normalized = img.astype(np.float32) / 255.0
        img_smooth = cv2.GaussianBlur(img_normalized, (5, 5), 1.0)

        return img_smooth

    def _calculate_cylinder_power(self, heightmap):
        grad_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)

        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)

        H = grad_xx + grad_yy
        K = grad_xx * grad_yy - grad_xy ** 2

        discriminant = np.sqrt(H ** 2 - 4 * K)
        k1 = (H + discriminant) / 2
        k2 = (H - discriminant) / 2

        scaling_factor = 337.5
        k1_diopters = k1 * scaling_factor
        k2_diopters = k2 * scaling_factor

        central_region = heightmap.shape[0] // 4
        center_y, center_x = heightmap.shape[0] // 2, heightmap.shape[1] // 2

        k1_central = np.mean(k1_diopters[center_y - central_region:center_y + central_region,
                             center_x - central_region:center_x + central_region])
        k2_central = np.mean(k2_diopters[center_y - central_region:center_y + central_region,
                             center_x - central_region:center_x + central_region])

        cylinder_power = abs(k1_central - k2_central)

        return cylinder_power, k1_central, k2_central

    def _calculate_astigmatism_axis(self, heightmap):
        grad_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)

        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)

        central_region = heightmap.shape[0] // 4
        center_y, center_x = heightmap.shape[0] // 2, heightmap.shape[1] // 2

        central_xx = grad_xx[center_y - central_region:center_y + central_region,
                     center_x - central_region:center_x + central_region]
        central_yy = grad_yy[center_y - central_region:center_y + central_region,
                     center_x - central_region:center_x + central_region]
        central_xy = grad_xy[center_y - central_region:center_y + central_region,
                     center_x - central_region:center_x + central_region]

        axis_angle = 0.5 * np.arctan2(2 * np.mean(central_xy),
                                      np.mean(central_xx) - np.mean(central_yy))
        axis_degrees = np.degrees(axis_angle) % 180

        return axis_degrees

    def _classify_astigmatism_type(self, axis_angle):
        if 0 <= axis_angle <= 30 or 150 <= axis_angle <= 180:
            return "With-the-rule (WTR)"
        elif 60 <= axis_angle <= 120:
            return "Against-the-rule (ATR)"
        else:
            return "Oblique"

    def _assess_regularity(self, heightmap):
        grad_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)

        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)

        mean_curvature = (grad_xx + grad_yy) / 2
        curvature_std = np.std(mean_curvature)

        regularity_threshold = 0.1

        if curvature_std < regularity_threshold:
            return "Regular", curvature_std
        else:
            return "Irregular", curvature_std

    def _calculate_simk_values(self, heightmap):
        center_y, center_x = heightmap.shape[0] // 2, heightmap.shape[1] // 2
        central_radius = min(heightmap.shape) // 8

        y_min = max(0, center_y - central_radius)
        y_max = min(heightmap.shape[0], center_y + central_radius)
        x_min = max(0, center_x - central_radius)
        x_max = min(heightmap.shape[1], center_x + central_radius)

        central_region = heightmap[y_min:y_max, x_min:x_max]

        grad_x = cv2.Sobel(central_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(central_region, cv2.CV_64F, 0, 1, ksize=3)
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)

        avg_curvature = np.mean((grad_xx + grad_yy) / 2)

        scaling_factor = 337.5
        simk = avg_curvature * scaling_factor

        return abs(simk)

    def _calculate_irregularity_indices(self, heightmap):
        center_y, center_x = heightmap.shape[0] // 2, heightmap.shape[1] // 2

        upper_half = heightmap[:center_y, :]
        lower_half = np.flipud(heightmap[center_y:, :])
        left_half = heightmap[:, :center_x]
        right_half = np.fliplr(heightmap[:, center_x:])

        min_rows = min(upper_half.shape[0], lower_half.shape[0])
        min_cols = min(left_half.shape[1], right_half.shape[1])

        upper_half = upper_half[:min_rows, :]
        lower_half = lower_half[:min_rows, :]
        left_half = left_half[:, :min_cols]
        right_half = right_half[:, :min_cols]

        vertical_asymmetry = np.mean(np.abs(upper_half - lower_half))
        horizontal_asymmetry = np.mean(np.abs(left_half - right_half))

        sai = (vertical_asymmetry + horizontal_asymmetry) / 2

        smooth_surface = cv2.GaussianBlur(heightmap, (15, 15), 5.0)
        sri = np.std(heightmap - smooth_surface)

        return {
            "SAI": sai * 100,
            "SRI": sri * 100,
            "Vertical_Asymmetry": vertical_asymmetry * 100,
            "Horizontal_Asymmetry": horizontal_asymmetry * 100
        }

    def _calculate_corneal_topographic_astigmatism(self, heightmap):
        cylinder_power, k1, k2 = self._calculate_cylinder_power(heightmap)
        axis = self._calculate_astigmatism_axis(heightmap)

        axis_rad = np.radians(axis * 2)
        j0 = -cylinder_power / 2 * np.cos(axis_rad)
        j45 = -cylinder_power / 2 * np.sin(axis_rad)

        cort_magnitude = np.sqrt(j0 ** 2 + j45 ** 2)

        return {
            "CorT_Magnitude": cort_magnitude,
            "J0": j0,
            "J45": j45,
            "Axis": axis
        }

    def _estimate_posterior_astigmatism(self, heightmap):
        anterior_cylinder, k1, k2 = self._calculate_cylinder_power(heightmap)
        posterior_cylinder = anterior_cylinder * 0.1

        anterior_axis = self._calculate_astigmatism_axis(heightmap)
        posterior_axis = (anterior_axis + 90) % 180

        return {
            "Posterior_Cylinder": posterior_cylinder,
            "Posterior_Axis": posterior_axis,
            "Note": "Estimated - requires actual posterior elevation data"
        }

    def analyze_eye_image(self, image_path):
        if self.learn is None:
            raise ValueError("Model not loaded. Please check model initialization.")

        pred, pred_idx, probs = self.learn.predict(image_path)
        confidence = float(probs[pred_idx])

        heightmap = self._preprocess_heightmap(image_path)

        cylinder_power, k1, k2 = self._calculate_cylinder_power(heightmap)
        axis = self._calculate_astigmatism_axis(heightmap)
        astigmatism_type = self._classify_astigmatism_type(axis)
        regularity, irregularity_measure = self._assess_regularity(heightmap)
        simk = self._calculate_simk_values(heightmap)
        irregularity_indices = self._calculate_irregularity_indices(heightmap)
        cort_values = self._calculate_corneal_topographic_astigmatism(heightmap)
        posterior_values = self._estimate_posterior_astigmatism(heightmap)

        return {
            "ai_classification": {
                "prediction": str(pred),
                "confidence": confidence,
                "all_probabilities": {str(self.learn.dls.vocab[i]): float(prob)
                                      for i, prob in enumerate(probs)}
            },
            "cylinder_power": cylinder_power,
            "astigmatism_axis": axis,
            "astigmatism_type": astigmatism_type,
            "regularity": {
                "type": regularity,
                "irregularity_measure": irregularity_measure
            },
            "simk_values": {
                "average_keratometry": simk,
                "k1_central": k1,
                "k2_central": k2
            },
            "irregularity_indices": irregularity_indices,
            "corneal_topographic_astigmatism": cort_values,
            "posterior_corneal_analysis": posterior_values
        }


def main():
    analyzer = CornealAnalyzer()
    return analyzer


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  
    analyzer = main()
