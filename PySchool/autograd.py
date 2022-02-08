import torch
from torch.autograd import Variable

def start():
    x = Variable(torch.FloatTensor([1,2,3]), requires_grad=True)
    print(x)

    y = x*x     # y' = 2x
    print(y)

    z = y*y     # z = y² = x⁴, z' = 4x³

    print(z)

    z.backward(torch.tensor([1,1,1]), retain_graph=True)

    print(x.grad.data)

    w = y.mean() # w = 1/3 (x1² + x2² + x3²), w' = 1/3 (2x1 + 2x2 + 2x3)
    print(w)

    w.backward()
    # x.grad.data.zero_()

    print(x.grad)

    # No grad
    # x.required_grad_(False)
    # x.detach
    # with torch.no_grad():



x = torch.tensor([[1], [2]], dtype=torch.float32)
w1 = torch.rand(2,2, dtype=torch.float32, requires_grad=True)
w2 = torch.rand(2, dtype=torch.float32, requires_grad=True)
print(x)
print(w1)
print(w2)


for epoch in range(3):
    h = torch.mm(w1.T, x).squeeze()
    print(f"h: {h}")

    y = torch.dot(w2.T, h)
    print(f"y: {y}")

    y.backward()
    
    print(w1.grad)
    w1.grad.zero_()
    
    print(w2.grad)
    w2.grad.zero_()


optimiser = torch.optim.SGD(w1, w2, dtype=torch.float32, lr=1e-3)

for epoch in range(3):
    h = torch.mm(w1.T, x).squeeze()
    print(f"h: {h}")

    y = torch.dot(w2.T, h)
    print(f"y: {y}")

    optimiser.step()
    optimiser.zero_grad()








