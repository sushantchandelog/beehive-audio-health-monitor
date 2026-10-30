import torch
import torch.nn as nn



class HiveCNN(nn.Module):
    def __init__(self):
        super(HiveCNN , self).__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(1,16 ,kernel_size=3, stride=1 , padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride=2),
            
            # Block 2
            nn.Conv2d(16,32 , kernel_size= 3,stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),
            
            #block 3
            nn.Conv2d(32, 64 , kernel_size=3 , stride=1 , padding =1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride= 2),
            
            #block4
            nn.Conv2d(64 , 128 , kernel_size= 3, stride = 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 , 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x,1)
        logits = self.classifier(x)
        
        return logits