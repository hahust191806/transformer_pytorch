import torch 
import torch.nn as nn 


"""
    Lý do sử dụng layernorm thay vì batchnorm là bởi trong các task NLP độ dài của các sentence có thể thay đổi tùy ý và không cố định 
    Nên nếu dùng batchnorm thì sẽ khó khăn cho việc tính toán độ lệch chuẩn và trung bình. 
    -> sử dụng layernorm 
"""
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps 
        
    def forward(self, x):
        """
            -1 means last dimension - chuẩn hóa cho từng feature trong batch 
            x có kích thước (batch_size, channels, height, width) -> -1 tính theo chiều width: nghĩa là chuẩn hóa các features 
            keepdim sẽ giữ kích thước của mean có cùng kích thước với x, ngoại trừ chiều được tính, ở trên -1 nghĩa là tính theo width, 
            vậy đầu ra có kích thước (batch_size, channels, height, 1)
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbias=False, keepdim=True)
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta 
        return out # đầu ra có cùng kích thước với đầu vào nhưng đã được chuẩn hóa 