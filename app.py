import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

# --- Model Definitions (copy from your training code) ---

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

# --- Load Model ---
@st.cache_resource
def load_model():
    model = FusionModel(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load('fusion_model_state_dict.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()
class_names = ['coccidiosis', 'healthy']  # Adjust if your label order is different

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Streamlit UI ---
st.title("Chicken Disease Classifier")
st.write("Upload a chicken image (JPG/PNG) to predict: **Healthy** or **Coccidiosis**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs, _ = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
    st.success(f"Prediction: **{pred_class.capitalize()}**")
