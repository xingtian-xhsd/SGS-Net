import os
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
from itertools import cycle
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import label_binarize

#from model.HIFuse import HiFuse_Tiny
#from model.other_model import HiFuse_Tiny
#from model.new_model import HiFuse_Tiny
#from model.main_model import HiFuse_Tiny
#from model.Resnet import resnet50
#from model.Visual_transformer import vit_base_patch16_224_in21k
#from torchvision.models.swin_transformer import swin_b, swin_t, Swin_T_Weights
#from torchvision.models import convnext_base, resnet34, \
#   vit_b_16, vgg19, efficientnet_b3
#from torchvision.models import mobilenet_v2, efficientnet_b0,vit_b_32
#from model.UniFormer import uniformer_base
#from torchvision.models import mobilenet_v2, efficientnet_b0,vit_b_32
#from model.starnet import starnet_s1
#from model.SwinTransformer import SwimTransformer
#from model.new_test_model import HiFuse_Tiny
from model.new_test_model2 import HiFuse_Tiny
#from model.cjy_model1 import HiFuse_Tiny


num_classes = 8
image_size = 224

class_labels = ['0', '1', '2', '3', '4', '5', '6', '7']
DATASETS_ROOT = r"F:\code\Dataset\Multi-class texture analysis in colorectal cancer histology"
ROOT_TEST = os.path.join(DATASETS_ROOT, 'test')
#F:\code\Dataset\kvasir_dataset
#F:\code\Dataset\Multi-class texture analysis in colorectal cancer histology
#F:\code\Dataset\ISIC2018
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = HiFuse_Tiny(num_classes=num_classes).to(device)

# Load weights
PRETRAINED_ROOT = f'checkpoint/main_model_2025_07_27_15_44_39_checkpoint.pth'
#main_model_2024_11_06_00_37_42_checkpoint.pth  #main_model_2025_07_23_09_38_16_checkpoint.pth
# Data preprocessing and loading
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root=ROOT_TEST, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Do not shuffle test data

# Load trained model weights
checkpoint = torch.load(PRETRAINED_ROOT)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode, no gradient computation

true_labels = []
predicted_labels = []
all_outputs = []

# Prediction
with torch.no_grad():
    for index, data in enumerate(test_loader):
        inputs, labels = data

        inputs = inputs.to(device)

        outputs = model(inputs)
        all_outputs.append(outputs.cpu().numpy())

        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        print("\r{} processing [{}/{}]".format(model.__class__.__name__, index + 1, len(test_dataset)), end="")
    print("\nEnd Predict")

# Combine all outputs
all_outputs = np.concatenate(all_outputs, axis=0)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
mcc = matthews_corrcoef(true_labels, predicted_labels)  # Calculate MCC
kappa = cohen_kappa_score(true_labels, predicted_labels)  # Calculate Cohen's Kappa

# Binarize the output
true_labels_bin = label_binarize(true_labels, classes=range(num_classes))

# Calculate ROC curve and AUC
probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微观AUC
fpr_micro, tpr_micro, _ = roc_curve(true_labels_bin.ravel(), probs.ravel())
micro_auc = auc(fpr_micro, tpr_micro)

# 自定义标签
labels = {
    0: 'dyed_lifted_polyps',
    1: 'dyed_resection_margins',
    2: 'esophagitis',
    3: 'normal_cecum',
    4: 'normal_pylorus',
    5: 'normal_z_line',
    6: 'polyps',
    7: 'ulcerative_colitis'

    # 0: 'TUMOR',
    # 1: 'STROMA',
    # 2: 'COMPLEX',
    # 3: 'LYMPHO',
    # 4: 'DEBRIS',
    # 5: 'MUCOSA',
    # 6: 'ADIPOSE',
    # 7: 'EMPTY'
}
# Plot ROC curve and AUC
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.4f})'
                   ''.format(labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve and AUC for Each Class')
plt.legend(loc="lower right")
plt.show()

print(f"Confusion Matrix: {conf_matrix}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1_score: {f1}")
print(f"MCC: {mcc}")
print(f"Kappa: {kappa}")
# 输出AUC
print(f"Micro AUC: {micro_auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
