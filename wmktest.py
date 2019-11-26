import torch as t
a = t.tensor([2.0],requires_grad=True)
b = t.tensor([2.0],requires_grad=True)

c = a+b 
c.backward()
print(c)