import os

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from torchvision.models import resnet50

from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm



# Paths and settings

data_dir = "/home/shrek/Desktop/Astigmata/dataset"

num_classes = 4

batch_size = 32

num_epochs = 100

train_percent = 0.8

class_names = ['astigmata', 'normal', 'cataract', 'diabetic_retinopathy']



# Ensure outputs folder exists

os.makedirs("outputs", exist_ok=True)



# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Mixup Augmentation

def mixup_data(x, y, alpha=0.4):

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam



def mixup_criterion(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



# Transforms

train_transforms = transforms.Compose([

    transforms.Resize(256),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



test_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



# Dataset

full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

train_size = int(train_percent * len(full_dataset))

test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

test_dataset.dataset.transform = test_transforms



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Model definition

def CNN(num_classes=4, pretrained=True):

    model = resnet50(pretrained=pretrained)

    model.fc = nn.Sequential(

        nn.Dropout(0.5),

        nn.Linear(model.fc.in_features, num_classes)

    )

    return model



# Training and evaluation

if __name__ == "__main__":

    model = CNN(num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)



    # Metric storage

    train_losses, val_losses = [], []

    train_accuracies, val_accuracies = [], []

    train_precisions, val_precisions = [], []

    train_recalls, val_recalls = [], []

    train_f1s, val_f1s = [], []



    best_val_acc = 0.0



    for epoch in range(num_epochs):

        model.train()

        train_loss, correct, total = 0.0, 0, 0

        all_preds, all_labels = [], []



        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} : Training"):

            images, labels = images.to(device), labels.to(device)

            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)



            optimizer.zero_grad()

            outputs = model(mixed_images)

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss.backward()

            optimizer.step()



            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())



        train_losses.append(train_loss / len(train_loader.dataset))

        train_accuracies.append(correct / total)

        train_precisions.append(precision_score(all_labels, all_preds, average="macro", zero_division=1))

        train_recalls.append(recall_score(all_labels, all_preds, average="macro"))

        train_f1s.append(f1_score(all_labels, all_preds, average="macro"))



        # Validation

        model.eval()

        val_loss, correct, total = 0.0, 0, 0

        val_preds, val_labels = [], []



        with torch.no_grad():

            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} : Validation"):

                images, labels = images.to(device), labels.to(device)

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

        val_precisions.append(precision_score(val_labels, val_preds, average="macro", zero_division=1))

        val_recalls.append(recall_score(val_labels, val_preds, average="macro"))

        val_f1s.append(f1_score(val_labels, val_preds, average="macro"))



        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        print(f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        print(f"Train Precision: {train_precisions[-1]:.4f}, Val Precision: {val_precisions[-1]:.4f}")

        print(f"Train Recall: {train_recalls[-1]:.4f}, Val Recall: {val_recalls[-1]:.4f}")

        print(f"Train F1: {train_f1s[-1]:.4f}, Val F1: {val_f1s[-1]:.4f}")



        # Check if this is the best model

        if val_accuracies[-1] > best_val_acc:

            best_val_acc = val_accuracies[-1]

            torch.save(model.state_dict(), "outputs/best_model.pth")



        # Step the scheduler

        scheduler.step()



    # Evaluation on Test Set

    model.load_state_dict(torch.load("outputs/best_model.pth"))

    model.eval()



    test_loss, correct, total = 0.0, 0, 0

    test_preds, test_labels = [], []



    with torch.no_grad():

        for images, labels in tqdm(test_loader, desc="Test Evaluation"):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            test_preds.extend(predicted.cpu().numpy())

            test_labels.extend(labels.cpu().numpy())



    test_accuracy = correct / total

    test_precision = precision_score(test_labels, test_preds, average="macro", zero_division=1)

    test_recall = recall_score(test_labels, test_preds, average="macro")

    test_f1 = f1_score(test_labels, test_preds, average="macro")



    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Test Precision: {test_precision:.4f}")

    print(f"Test Recall: {test_recall:.4f}")

    print(f"Test F1: {test_f1:.4f}")



    # Confusion Matrix

    cm = confusion_matrix(test_labels, test_preds, labels=np.arange(num_classes))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap=plt.cm.Blues)

    plt.savefig('outputs/confusion_matrix.png')



    # Plot training and validation metrics

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)

    plt.plot(range(num_epochs), train_accuracies, label="Train Accuracy")

    plt.plot(range(num_epochs), val_accuracies, label="Validation Accuracy")

    plt.title('Accuracy over Epochs')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()



    plt.subplot(1, 2, 2)

    plt.plot(range(num_epochs), train_f1s, label="Train F1 Score")

    plt.plot(range(num_epochs), val_f1s, label="Validation F1 Score")

    plt.title('F1 Score over Epochs')

    plt.xlabel('Epochs')

    plt.ylabel('F1 Score')

    plt.legend()



    plt.tight_layout()

    plt.savefig('outputs/metrics_over_epochs.png')

    plt.show()