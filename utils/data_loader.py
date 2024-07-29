from torchtext.legacy.data import Field, BucketIterator, Iterator
from torchtext.legacy.datasets.translation import Multi30k 

from tokenizer import Tokenizer


class DataLoader(): 
    source: Field = None 
    target: Field = None 
    
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext # chứa danh sách các phần mở rộng của tệp cho các ngôn ngữ. VD: ('.de', '.en')
        self.tokenize_en = tokenize_en 
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        """
            thuộc tính batch_first=True trong lớp Field của torchtext chỉ định rằng dữ liệu đầu vào sẽ được sắp xếp theo thứ tự 
            [batch_size, sequence_length] thay vì [sequence_length, batch_size]. 
        """
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token, 
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        
        return train_data, valid_data, test_data
    
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)
        
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
    
if __name__ == "__main__":
    ext = ('.de', '.en')
    tokenizer = Tokenizer()
    init_token='<sos>'
    eos_token='<eos>'
    loader = DataLoader(ext=ext, 
                        tokenize_de=tokenizer.tokenize_de, 
                        tokenize_en=tokenizer.tokenize_en, 
                        init_token=init_token, 
                        eos_token=eos_token)
    
    train, valid, test = train, valid, test = loader.make_dataset()