import spacy 


class Tokenizer: 
    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]
    
    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
    
if __name__ == "__main__":
    ext = ('.de', '.en')
    tokenizer = Tokenizer()
    init_token='<sos>'
    eos_token='<eos>'
    # Khởi tạo tokenizer cho các ngôn ngữ
    tokenize_en = tokenizer.tokenize_en("Tokenizes English text from a string into a list of strings")
    print(tokenize_en)
    tokenize_de = tokenizer.tokenize_de("Tokenisiert englischen Text aus einer Zeichenfolge in eine Liste von Zeichenfolgen")
    print(tokenize_de)