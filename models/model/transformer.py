import torch
import torch.nn as nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    """
        max_len: đại diện cho độ dài tối đa mà một sentences có thể có 
        enc_voc_size: kích thước của từ điển của encoder 
        dec_voc_size: kích thước của từ điển của dencoder
    """
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx # index của padding trong sentences soource 
        self.trg_pad_idx = trg_pad_idx # index của padding trong sentences target 
        self.trg_sos_idx = trg_sos_idx # index của kí tự đánh dấu là bắt đầu sentences targetf
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    """
        Đầu vào của transformer sẽ là các batch, chứa các tensor đại diện cho các sentences, mỗi giá trị trong tensor là index của token trong từ điển được tạo
        Giá trị 0 đại diện cho padding được thêm vào để batch size có kích thước bằng nhau
    """
    def forward(self, src, trg):
        """
            src: tensor [num_sample, seq_len]
            trg: tensor [num_sample, seq_len]
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    # tạo mask cho input đầu vào, với source sentence, mask chính là các vị trí padding được thêm vào để đảm bảo các tensor trong batch có độ dài bằng nhau
    def make_src_mask(self, src): # -> trg [num_of_sample, d_model]
        """
            # Ví dụ tensor trg và chỉ số padding
            trg = torch.tensor([
                [1, 2, 3, 0],  # Câu 1 (0 là padding)
                [4, 5, 6, 0]   # Câu 2 (0 là padding)
            ]) - (2, 4)
            -> output: src_mask - (2, 1, 1, 4)
            tensor([[[[ True,  True,  True, False]]],
                    [[[ True,  True,  True, False]]])
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # -> [num_of_sample, 1, seq_len, 1]
        return src_mask # trả về một tensor chứa giá trị boolean, .unsqueeze(1).unsqueeze(2) mở rộng kích thước tensor để phù hợp với model 

    def make_trg_mask(self, trg): # -> trg [num_of_sample, seq_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3) # -> [num_of_sample, 1, 1, seq_len]
        """
            tensor([[1, 2, 0, 4],
                    [5, 0, 0, 7]])
            -> tensor([[[[True, True, False, True]],
                    [[True, False, False, True]]]])
        """
        trg_len = trg.shape[1] # -> num_of_sample = length of sentences
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device) # -> torch.tril tạo ma trận tam giác từ ma trận vuông khởi tạo, giá trị dưới đường chéo chính giữ nguyên 1, còn lại đổi thành 0 
        """
            tensor([[1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1]], dtype=torch.uint8)
                    
            ->tensor([[[[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 1, 1, 1]]],

                        [[[1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 1, 1, 1]]]], dtype=torch.uint8)
        """
        trg_mask = trg_pad_mask & trg_sub_mask 
        return trg_mask # trả về một tensor chứa giá trị boolean, chỉ những vị trí mask và pad [num_of_sample, 1, seq_len, seq_len]