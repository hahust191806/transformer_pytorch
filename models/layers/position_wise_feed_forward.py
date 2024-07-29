import torch.nn as nn 


class PositionwiseFeedForward(nn.Module):
    """
        Nhận đầu vào là một tensor có kích thước (batch_size, sequence_length, d_model)
        nn.Linear áp dụng phép biến đổi tuyến tính cho từng hàng của ma trận đầu vào một cách độc lập. 
        Example: 
        - Đầu vào có kích thước: (batch_size, sequence_length, d_model)
        -> Đầu ra cũng sẽ có kích thước (batch_size, sequence_length, d_model)
    """
    
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x # return a tensor that have same size with input tensor 