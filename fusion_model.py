import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

# --- Model Definitions (copy these from your training code) ---

class ResNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetModel, self).__init__()
        from torchvision import models
        self.resnet = models.resnet50(pretrained=pretrained)
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
    def forward(self, x):
        features = self.resnet(x)
        output = self.fc(features)
        return output, features

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_name='efficientnet-b0'):
        super(EfficientNetModel, self).__init__()
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained(model_name)
        else:
            self.efficientnet = EfficientNet.from_name(model_name)
        self.feature_dim = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
    def forward(self, x):
        features = self.efficientnet(x)
        output = self.fc(features)
        return output, features

class FusionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(FusionModel, self).__init__()
        self.resnet_model = ResNetModel(num_classes, pretrained)
        self.efficientnet_model = EfficientNetModel(num_classes, pretrained)
        self.resnet_features = self.resnet_model.feature_dim
        self.efficientnet_features = self.efficientnet_model.feature_dim
        self.combined_dim = self.resnet_features + self.efficientnet_features
        self.attention = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        _, resnet_features = self.resnet_model(x)
        _, efficientnet_features = self.efficientnet_model(x)
        combined_features = torch.cat((resnet_features, efficientnet_features), dim=1)
        attention_weights = self.attention(combined_features)
        weighted_features = combined_features * attention_weights
        output = self.fusion(weighted_features)
        return output, combined_features

# --- End Model Definitions ---

# --- Load Model ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2  # Adjust if needed

# Option 1: Load from state_dict (recommended)
model = FusionModel(num_classes=num_classes, pretrained=False)
state_dict = torch.load('fusion_model_state_dict.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# # Option 2: Load entire model (if you saved with torch.save(model, ...))
# model = torch.load('fusion_model.pth', map_location=device)
# model.to(device)
# model.eval()

# --- Image Preprocessing ---

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Class Names ---
class_names = ['coccidiosis', 'healthy']  # Adjust to match your label order

# --- Prediction Function ---

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, _ = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
        print(f"Image: {os.path.basename(image_path)} --> Predicted: {pred_class}")
    return pred_class

# --- Test Individual Images ---

if __name__ == "__main__":
    # Example: test a single image
    test_image_path = "healthy1.JPG"  # Change to your image path
    predict_image(test_image_path)

    # Or test multiple images in a folder
    # test_folder = "test_images"
    # for fname in os.listdir(test_folder):     
    #     if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
    #         predict_image(os.path.join(test_folder, fname))
