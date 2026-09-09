import torch
import torch.nn as nn

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[1.]])

model = nn.Linear(2,1)
sigmoid = nn.Sigmoid()

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    z = model(X)
    prediction = sigmoid(z)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    for i in range(len(X)):
        z = model(X[i])
        pred = sigmoid(z)
        print(f"Input: {X[i].tolist()}, Predicted:{pred.item():.4f}, Actual: {y[i].item()}")