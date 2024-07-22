import torch 
import torch.nn as nn 


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps 
        
    def forward(self, x):
        """
            -1 means last dimension 
            x có kích thước (batch_size, channels, height, width) -> -1 tính theo chiều width: nghĩa là chuẩn hóa các features 
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbias=False, keepdim=True)
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta 
        return out 