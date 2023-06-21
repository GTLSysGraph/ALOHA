import torch
# a= torch.tensor([[1,0,1,1,0,0,1],[1,0,1,1,0,0,1]])
# b= torch.tensor([[0,1,1,0,1,1,1],[1,0,1,1,0,0,1]])
# intersection = torch.count_nonzero(a * b,dim =1)
# J = intersection * 1.0 / (torch.count_nonzero(a,dim=1) + torch.count_nonzero(b,dim=1) - intersection)




a= torch.tensor([45,213,12,1,30,210,18])

print(torch.where(a>=45))
print(torch.where(a<45))
