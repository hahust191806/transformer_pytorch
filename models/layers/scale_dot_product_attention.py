import math 
import torch.nn as nn 


class ScaleDotProductAttention(nn.Module):
    """
        compute scale dot product attention 
        
        Query: given sentence that we focused on (decoder)
        Key: every sentence to check relationship with Query (encoder)
        Value: every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size() 
        
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3) # transpose 
        score = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product 

        # 2. apply masking (opt) => mask is a tensor that has same size of score matrix
        if mask is not None: 
            score = score.masked_fill(mask == 0, -10000)
        """
            score = torch.tensor([[0.1, 0.3, 0.2],
                      [0.5, 0.1, 0.4]])
            mask = torch.tensor([[1, 0, 1],
                     [1, 1, 0]])
            masked_score = score.masked_fill(mask == 0, -10000)
            # tensor([[0.1, -10000.0, 0.2],
            #         [0.5, 0.1, -10000.0]])
        """

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v 
        
        return v, score 