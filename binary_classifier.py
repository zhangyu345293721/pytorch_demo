
import torch
import torch.nn as nn
import torch.optim as optim

# use pytorch achieve binary search classifier

# data preparation
X = torch.tensor([[2.0, 1.0], [3.0, 2.0], [3.0, 4.0], [4.0, 5.0]])
y = torch.tensor([0, 0, 1, 1])

# define model
class LinearClassifier(nn.Module):
    """
        classifier model
    """
    def __init__(self, input_size : int , output_size : int) -> None:
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x) -> object:
        return self.linear(x)

# init model and model parameter init 
input_size = 2  
output_size = 2  
model = LinearClassifier(input_size, output_size)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)  

# epoch train model
num_epochs = 100
for epoch in range(num_epochs):
    """
        train model and pringt epoch loss
    """
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# test model predict
with torch.no_grad():
    """
        use model prediction
    """
    outputs = model(X)
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    print(predicted)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    print('acc: {:.2f}%'.format(correct / total * 100))
