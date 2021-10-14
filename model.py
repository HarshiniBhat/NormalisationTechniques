import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self,drop = 0.05, norm = 'bn' , groups = 4):
    
    super(Net, self).__init__()
    self.max_pool = nn.MaxPool2d(2,2)
    def Normalize(x):
      if norm == 'bn':
        return nn.BatchNorm2d(x)
      elif norm =='ln':
        return nn.GroupNorm(1,x)
      elif norm =='gn':
        return nn.GroupNorm(groups,x)
      else:
        return None


    

    self.conv_block1 = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1, bias=False),
        Normalize(8),
        nn.ReLU(),
        nn.Dropout2d(drop),

        nn.Conv2d(8, 24, 3, padding=1, bias=False),
        Normalize(24),
        nn.ReLU(),
        nn.Dropout2d(drop),

       
    )

    self.transition1 = nn.Sequential(
        nn.Conv2d(24, 8, 1, bias=False),
        Normalize(8),
        nn.ReLU(),
        nn.Dropout2d(drop),
    )

    self.conv_block2 = nn.Sequential(
        nn.Conv2d(8, 16, 3, bias=False),
        Normalize(16),
        nn.ReLU(),
        nn.Dropout2d(drop),

        nn.Conv2d(16, 24, 3, bias=False),
        Normalize(24),
        nn.ReLU(),
        nn.Dropout2d(drop),
    )

    self.transition2 = nn.Sequential(
        nn.Conv2d(24, 8, 1, bias=False),
        Normalize(8),
        nn.ReLU(),
        nn.Dropout2d(drop),
    )
  
    self.conv_block3 = nn.Sequential(
        nn.Conv2d(8, 16, 3, bias=False),
        Normalize(16),
        nn.ReLU(),
        nn.Dropout2d(drop),

        nn.Conv2d(16, 32, 3, bias=False),
        Normalize(32),
        nn.ReLU(),
        nn.Dropout2d(drop),
    )

    self.gap = nn.Sequential(
        nn.AvgPool2d(6)
    )
    self.dense = nn.Linear(32, 10)

  def forward(self, x):
    x = self.conv_block1(x)
    x = self.max_pool(x)
    x = self.transition1(x)

    x = self.conv_block2(x)
    x = self.transition2(x)

    x = self.conv_block3(x)

    x = self.gap(x)
    x = x.view(-1, 32)    
    x = self.dense(x)

    return F.log_softmax(x, dim=1)