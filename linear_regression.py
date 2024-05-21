import torch
import torch.nn as nn
import torch.optim as optim

# use pytorch implement linear regression
# preparation data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# define linear model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# init model and optimizer
input_size = 1  # input featrue
output_size = 1  # output feature
model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss()  # mse
optimizer = optim.SGD(model.parameters(), lr=0.01)  # sgd

# train model
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# test model
with torch.no_grad():
    predicted = model(X)
    print('prediction:', predicted.numpy())
