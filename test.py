from torch import nn
from torchvision.models import vit_b_16, densenet121

model = densenet121()
model.fc = nn.Linear(2048, 8)
print(model)