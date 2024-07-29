import os 
import dill as pickle 
import pandas as pd 
import re 
import spacy
from torchtext import data 


# class mã hóa chuỗi đầu vào 
class tokenize(object):
    def __init__(self, lang): 
        self.nlp = spacy.load(lang) # sử dụng load method của spacy để tạo mô hình mã hóa token
        
    """
        Nếu phần tử khác " " thì mã hóa bằng self.nlp.tokenizer 
        -> Trả về 1 list các token đã được chuẩn hóa
    """
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
            

# hàm đọc dữ liệu, đầu vào là path đến các file chứa samples source - en và target - vn 
def read_data(src_file, trg_file):
    # đọc file và loại bỏ khoảng trắng, sau đó split theo kí tự xuống dòng 
    src_data = open(src_file).read().strip().split('\n')
    trg_data = open(trg_file).read().strip().split('\n')
    
    return src_data, trg_data # trả về một list các string là các sample 

# hàm tạo các field để preprocess data ~ transforms trong pytorch
def create_fields(src_lang, trg_lang): # đầu vào là các string chứa giá trị ngôn ngữ 
    print("loading spacy tokenizers...")

    # khởi tạo đối tượng mã hóa chuỗi đầu vào 
    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)
    
    # khởi tạo các Field gồm các bước tiền xử lý 
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
    
    return SRC, TRG # trả về 2 đối tượng Field Objects 

# hàm tạo đối tượng data loader 
def create_dataset(src_data, trg_data, max_strlen, batchsize, device, SRC, TRG, istrain=True):
    
    print("creating dataset and iterator... ")
    
    # tạo một dictionary chứa 2 keys là src và trg, mỗi key có value chính là một list các samples
    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"]) # khởi tạo một dataframe gồm 2 cột

    # tạo mask cho dataset để lọc ra các setences có độ dài nhỏ hơn max_len 
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]
    
    df.to_csv("translate_transformer_temp.csv", index=False)
    
    # khởi tạo một dataset object từ một định dạng bảng như CSV, TSV hoặc các file tương tự. 
    data_fields = [('src', SRC), ('trg', TRG)] # khởi tạo 1 list gồm các tuple chứa tên columns và Field tương ứng 
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    
    # os.remove('translate_transformer_temp.csv')