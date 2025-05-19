import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device, class_names=None):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n✅ Final Evaluation on Test Set:")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save metrics to outputs folder
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/final_metrics.txt", "w") as f:
        f.write(f"Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Confusion matrix heatmap
    if class_names:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("outputs/confusion_matrix.png")
        plt.close()

    return accuracy, precision, recall, f1, cm


# 🔁 Standalone usage
if __name__ == "__main__":
    # Settings
    data_dir = "/home/shrek/Desktop/Astigmata/dataset"
    num_classes = 4
    batch_size = 32
    model_path = "outputs/best_model.pth"
    class_names = ['astigmata', 'normal', 'catarct', 'diabetic_retinopathy']

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    test_size = int(0.2 * len(full_dataset))
    _ = len(full_dataset) - test_size
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [_, test_size], generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Run evaluation
    evaluate_model(model, test_loader, device, class_names)
