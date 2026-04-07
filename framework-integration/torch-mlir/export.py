import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through ``fc1``
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

# Equates to one random 28x28 image
random_data = torch.rand((1, 1, 28, 28))

my_nn = Net()
result = my_nn(random_data)


module = fx.export_and_import(
    model,
    random_data,
    output_type="linalg-on-tensors",
)

with open("result.txt", "w") as f: 
    f.write(module.operation.get_asm())