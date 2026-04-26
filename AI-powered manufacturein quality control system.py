import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/train"   # dataset folder
MODEL_PATH = "model.pth"

CLASSES = ["good", "scratch", "dent", "color_variation", "dimension_issue"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL
# -----------------------------
class DefectModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# DATASET
# -----------------------------
class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, cls in enumerate(CLASSES):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.exists(cls_path):
                continue
            for img in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = DefectDataset(DATA_PATH, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DefectModel(len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")

    for epoch in range(5):
        total_loss = 0
        model.train()

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

# -----------------------------
# PREDICTION
# -----------------------------
def predict_image(frame, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    return CLASSES[pred.item()], confidence.item()

# -----------------------------
# REAL-TIME CAMERA
# -----------------------------
def run_realtime():
    model = DefectModel(len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_image(frame, model)

        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("AI Quality Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# MAIN MENU
# -----------------------------
if __name__ == "__main__":
    print("\n1. Train Model")
    print("2. Run Real-time Detection")

    choice = input("Enter choice: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        run_realtime()
    else:
        print("Invalid choice")