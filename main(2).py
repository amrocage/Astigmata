import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Import LambdaLR for combined scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import math # Import math for calculating warmup LR

# Set Paths and Settings
data_dir = "/home/shrek/Desktop/Astigmata/dataset"
num_classes = 4
batch_size = 32
num_epochs = 120
train_percent = 0.8
class_names = ['astigmata', 'normal', 'cataract', 'diabetic_retinopathy']
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Learning Rate Warmup Settings
warmup_epochs = 5 # Number of epochs for linear warmup
warmup_lr_factor = 0.1 # Starting LR factor (e.g., 0.1 * base_lr)

# Image Resolution Settings
# Increase the input resolution for potentially better accuracy
IMG_RESIZE_SIZE = 288 # Resize images to this size
IMG_CROP_SIZE = 256 # Crop/evaluate at this size

# Mixup Function
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# SAM Optimizer (Sharpness-Aware Minimization)
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        # Ensure params is a list of param_groups or convert it
        self.param_groups = list(params) if isinstance(params, (list, tuple)) and isinstance(params[0], dict) else [{'params': list(params)}]

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho
        self.adaptive = adaptive
        self.defaults = self.base_optimizer.defaults
        self.state = {} # Initialize the state dictionary

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                # Ensure parameter is hashable and requires grad
                if p.requires_grad and p not in self.state:
                    self.state[p] = {}

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group['params']:
                # Only process parameters with gradients and that require grad
                if p.grad is None or not p.requires_grad: continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                # Store e_w in the state dictionary for this parameter
                # Ensure the parameter is in the state dict
                if p in self.state:
                   self.state[p]['e_w'] = e_w


        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                 # Only process parameters that require grad
                if not p.requires_grad: continue
                # Ensure the parameter and 'e_w' are in the state before subtracting
                if p in self.state and 'e_w' in self.state[p]:
                     p.sub_(self.state[p]['e_w'])
                     # Optional: Clean up the state after use to save memory
                     # del self.state[p]['e_w']
                else:
                    # This can happen for parameters that didn't get gradients in the first step (e.g., frozen layers)
                    # Or if first_step was skipped for some reason.
                    # print(f"Warning: 'e_w' not found in state for parameter {p.shape if hasattr(p, 'shape') else 'N/A'}")
                    pass


        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    # The step method for SAM when NOT using GradScaler (AMP)
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        self.zero_grad()
        loss = closure()
        self.first_step(zero_grad=False)
        self.zero_grad()
        closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        # Only consider parameters with gradients and that require grad
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.requires_grad and hasattr(p, 'grad') and p.grad is not None
            ]),
            p=2
        )
        return norm

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.base_optimizer.add_param_group(param_group)

# Data Augmentation and Preprocessing
train_transforms = transforms.Compose([
    transforms.Resize(IMG_RESIZE_SIZE), # Increased resize size
    transforms.RandomResizedCrop(IMG_CROP_SIZE), # Increased crop size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), # Slightly increased rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15), # Slightly stronger color jitter
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10), # Added RandomAffine
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_CROP_SIZE, IMG_CROP_SIZE)), # Resize to evaluation size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and Dataloader
# Load the full dataset with training transforms initially
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

# Split the dataset into train and test subsets
train_size = int(train_percent * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply test transforms specifically to the test dataset Subset
# This is the correct way to set transforms for the test split after random_split
test_dataset.transform = test_transforms # Corrected line


# Consider increasing num_workers for faster data loading if CPU isn't a bottleneck
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


# Model Definition
def build_model():
    # dropout is already included via drop_rate
    model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=num_classes, drop_rate=0.3)
    return model

# Training Loop
if __name__ == "__main__":
    model = build_model().to(device)

    base_optimizer = optim.AdamW
    # Initial LR is the maximum LR after warmup
    optimizer = SAM(model.parameters(), base_optimizer, lr=3e-4, weight_decay=1e-4)

    # Define the combined scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: LR goes from warmup_lr_factor * base_lr to base_lr
            return warmup_lr_factor + (1.0 - warmup_lr_factor) * epoch / warmup_epochs
        else:
            # Cosine annealing after warmup: LR goes from base_lr to 0
            cosine_epoch = epoch - warmup_epochs
            cosine_max_epochs = num_epochs - warmup_epochs
            if cosine_max_epochs <= 0: return 0.0 # Handle case where warmup is >= num_epochs
            return 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_max_epochs))

    # LambdaLR applies the lr_lambda function to the initial learning rate of the optimizer
    scheduler = LambdaLR(optimizer.base_optimizer, lr_lambda)


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Initialize EMA model
    # EMA can help stabilize the final model's performance
    ema_model = AveragedModel(model)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = [] # List to store learning rates per epoch
    best_val_acc = 0.0

    # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
    # This warning is expected when stepping the scheduler after the epoch,
    # as optimizer.step() happens per batch. It's generally safe to ignore
    # for epoch-wise schedulers like this, as it only affects the very first LR value.
    # To strictly remove it, you could step the scheduler manually after the very first
    # optimizer step in the first batch, or use a different scheduler.
    # We will keep stepping after the epoch for simplicity with epoch-based schedulers.


    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        # Get and store the current learning rate
        current_lr = optimizer.base_optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)


        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} : Training")

        for images, labels in pbar:
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Define the closure function required by SAM
            # This function performs the forward and backward pass
            def closure():
                # Gradients are zeroed by optimizer.step()
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                # Perform the backward pass within the closure
                loss.backward()

                # Return the unscaled loss
                return loss

            # Call the SAM optimizer step with the closure
            # The SAM step method handles the two passes and base optimizer update
            loss = optimizer.step(closure)

            # Calculate training metrics *after* the SAM step updates the weights
            # Use the unscaled loss from the first pass for reporting
            train_loss += loss.item() * images.size(0)

            with torch.no_grad():
                # Evaluate on original images for accurate training metric after weight update
                # Briefly set to eval mode if needed for metric calculation (e.g., for dropout/BN)
                model.eval()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                model.train() # Set back to train mode


        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracies.append(correct / total)

        # Validation
        model.eval() # Set model to evaluation mode
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} : Validation"):
                # Move data to device
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(test_loader.dataset))
        val_accuracies.append(correct / total)

        # EMA update
        # Update the EMA model parameters based on the current model parameters
        ema_model.update_parameters(model)

        # Print Stats
        print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracies[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Accuracy: {val_accuracies[-1]:.4f}, "
              f"Current LR: {current_lr:.6f}")


        # Save best model based on validation accuracy
        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            # Save the state_dict of the actual model
            torch.save(model.state_dict(), "outputs/best_model.pth")

        # Step the learning rate scheduler based on the epoch
        # This happens after all optimizer steps for the epoch are done
        scheduler.step()


    print("Training Complete!")

    # --- Final Evaluation and Metrics ---

    # Evaluate the best model saved during training
    print("\nEvaluating Best Model...")
    best_model = build_model().to(device)
    # Load the state dict of the best performing model
    best_model.load_state_dict(torch.load("outputs/best_model.pth"))
    best_model.eval() # Set to evaluation mode

    final_test_loss, final_correct, final_total = 0.0, 0, 0
    final_preds, final_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Evaluation (Best Model)"):
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = best_model(images)
            loss = criterion(outputs, labels)
            final_test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
            final_preds.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    final_test_loss /= len(test_loader.dataset)
    final_test_accuracy = final_correct / final_total
    final_precision = precision_score(final_labels, final_preds, average='weighted')
    final_recall = recall_score(final_labels, final_preds, average='weighted')
    final_f1 = f1_score(final_labels, final_preds, average='weighted')
    final_cm = confusion_matrix(final_labels, final_preds)

    print(f"\nFinal Evaluation (Best Model):")
    print(f"  Loss: {final_test_loss:.4f}")
    print(f"  Accuracy: {final_test_accuracy:.4f}")
    print(f"  Precision (weighted): {final_precision:.4f}")
    print(f"  Recall (weighted): {final_recall:.4f}")
    print(f"  F1-Score (weighted): {final_f1:.4f}")
    print("  Confusion Matrix:")
    print(final_cm)

    # Plotting the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Best Model)")
    plt.savefig("outputs/confusion_matrix_best_model.png")
    plt.close()


    # SWA (Stochastic Weight Averaging) - After training loop
    print("\nUpdating SWA model BatchNorm statistics...")
    # Initialize SWA model with the final trained model weights
    swa_model = AveragedModel(model)
    # Update BN for SWA model using the training data
    swa_model.train() # Set SWA model to train mode for BN update
    if hasattr(swa_model, 'module'): # Check if model is wrapped (e.g., DataParallel)
        update_bn(train_loader, swa_model.module, device=device)
        # Save the state_dict of the module after updating BN
        torch.save(swa_model.module.state_dict(), "outputs/swa_model_with_bn_updated.pth")
    else:
        update_bn(train_loader, swa_model, device=device)
        # Save the state_dict of the AveragedModel after updating BN
        torch.save(swa_model.state_dict(), "outputs/swa_model_with_bn_updated.pth")
    print("SWA BatchNorm update complete.")


    # Evaluate the SWA model
    print("\nEvaluating SWA Model...")
    # Load the SWA model state dictionary with updated BN statistics
    swa_model_eval = build_model().to(device)
    # Load the state dict from the saved file after update_bn
    # This loads the averaged weights and updated BN stats
    swa_model_eval.load_state_dict(torch.load("outputs/swa_model_with_bn_updated.pth"))
    swa_model_eval.eval() # Set to evaluation mode

    swa_test_loss, swa_correct, swa_total = 0.0, 0, 0
    swa_preds, swa_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Evaluation (SWA Model)"):
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = swa_model_eval(images)
            loss = criterion(outputs, labels)
            swa_test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            swa_total += labels.size(0)
            swa_correct += (predicted == labels).sum().item()
            swa_preds.extend(predicted.cpu().numpy())
            swa_labels.extend(labels.cpu().numpy())

    swa_test_loss /= len(test_loader.dataset)
    swa_test_accuracy = swa_correct / swa_total
    swa_precision = precision_score(swa_labels, swa_preds, average='weighted')
    swa_recall = recall_score(swa_labels, swa_preds, average='weighted')
    swa_f1 = f1_score(swa_labels, swa_preds, average='weighted')
    swa_cm = confusion_matrix(swa_labels, swa_preds)

    print(f"\nFinal Evaluation (SWA Model):")
    print(f"  Loss: {swa_test_loss:.4f}")
    print(f"  Accuracy: {swa_test_accuracy:.4f}")
    print(f"  Precision (weighted): {swa_precision:.4f}")
    print(f"  Recall (weighted): {swa_recall:.4f}")
    print(f"  F1-Score (weighted): {swa_f1:.4f}")
    print("  Confusion Matrix:")
    print(swa_cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=swa_cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (SWA Model)")
    plt.savefig("outputs/confusion_matrix_swa_model.png")
    plt.close()


    # Plotting Loss, Accuracy, and Learning Rate curves
    plt.figure(figsize=(18, 5)) # Increased figure size for three plots
    plt.subplot(1, 3, 1) # Changed to 1 row, 3 columns
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2) # Changed column index
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3) # Added new subplot for Learning Rate
    plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig("outputs/loss_accuracy_lr_plots.png")
    plt.close()