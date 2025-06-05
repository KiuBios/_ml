import torch
import torch.nn as nn
import torch.optim as optim

seven_segment_inputs = torch.tensor([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1],  # 9
], dtype=torch.float32)

binary_outputs = torch.tensor([
    [0,0,0,0],  # 0
    [0,0,0,1],  # 1
    [0,0,1,0],  # 2
    [0,0,1,1],  # 3
    [0,1,0,0],  # 4
    [0,1,0,1],  # 5
    [0,1,1,0],  # 6
    [0,1,1,1],  # 7
    [1,0,0,0],  # 8
    [1,0,0,1],  # 9
], dtype=torch.float32)

class SegmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 16)
        self.layer2 = nn.Linear(16, 4)
        self.act = nn.Sigmoid()  # 輸出範圍限制在 0~1

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.act(self.layer2(x))
        return x

model = SegmentModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    output = model(seven_segment_inputs)
    loss = loss_fn(output, binary_outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def predict(input_tensor):
    with torch.no_grad():
        pred = model(input_tensor)
        pred_binary = (pred > 0.5).int()
    return pred_binary

print("\n測試預測結果：")
for i in range(10):
    input_vec = seven_segment_inputs[i]
    expected = binary_outputs[i].int().tolist()
    predicted = predict(input_vec.unsqueeze(0)).squeeze().tolist()
    print(f"Input {i}: 預測 = {predicted}, 正確 = {expected}")
