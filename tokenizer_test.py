import pandas as pd
from transformers import PreTrainedTokenizerFast

"""
cls token 이 하는 일은 classification 임.
"""


class TestTokenizer:

    def __init__(self, filepath, tok_vocab, max_seq_len=128):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_vocab,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>',
            cls_token='<cls>'
        )

    def make_input_id_mask(self, tokens):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        print(input_id)
        print(self.tokenizer.eos_token_id)
        print(self.tokenizer.cls_token_id)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask

    def tokenize(self, index):
        record = self.data.iloc[index]
        q, a, label = record['Q'], record['A'], record['label']
        tokenized = self.tokenizer.tokenize(q)
        print(tokenized)
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(tokenized)


        # for i in range(len(self.data) - 1):
        #     tokenized = self.tokenizer.tokenize(self.data.iloc[i + 1]['Q'])
        #     if tokenized.count("<unk>") > 0 or tokenized.count("<pad>") > 0 or tokenized.count("<mask>") > 0:
        #         print(tokenized)


if __name__ == "__main__":
    tokenizer = TestTokenizer(
        filepath='./refined_sentimental/train.csv',
        tok_vocab="./emji_tokenizer/model.json"
    )
    tokenizer.tokenize(1000)

