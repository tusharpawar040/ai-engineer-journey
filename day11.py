import torch
import torch.nn as nn

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

# model
model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)

# training
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(2000):
    prediction = model(X)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    for i in range(len(X)):
        pred = model(X[i])
        print(f"Input: {X[i].tolist()}, Predicted: {pred.item()}, Actual: {y[i].item()}")