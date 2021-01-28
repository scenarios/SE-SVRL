import torch

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.l = torch.nn.Linear(3,1)
        self.q = torch.nn.Linear(3,1)
        self.mdict = torch.nn.ModuleDict({'l':self.l, 'q': self.q})
        del self.mdict['q'].weight
    def forward(self, x):
        self.q.weight = torch.mean(self.l.weight, axis=0, keepdim=True)
        self.q.weight.retain_grad()
        return self.mdict['q'](x)

net = NN()
for param in net.parameters():
    print(param.data)
print('###############')
net(torch.randn(3)).backward()
for param in net.parameters():
    print(param.data)

print('###############')
print(net.q.weight.grad)
print(net.l.weight.grad)

net(torch.randn(3)).backward()

print('###############')
print(net.q.weight.grad)
print(net.l.weight.grad)


