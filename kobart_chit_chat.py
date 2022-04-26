import argparse
import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
# from transformers import (BartForConditionalGeneration,
#                           PreTrainedTokenizerFast)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

"""
처음 프로젝트를 init 할 때, 명령어대로 입력해야 tokenizer 가 다운로드 되는 구조입니다.

from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

이후
get_kobart_tokenizer(".") 
를 실행하면
1. kobart_base_tokenizer_cased_xxx.....zip 이 생성됩니다.
    아마 파라미터로 캐시 폴더를 입력받는 것 같고, "." 를 쓰게 되면 현재 폴더를 지정합니다.
2. emji_tokenizer 라는 폴더가 생성됩니다.
해당 파일은 토크나이저 압축 파일, emji_tokenizer 는 토크나이저의 압축을 푼 원본 모델이라고 생각됩니다. 
실제 모델은 emji_tokenizer/model.json 이라고 추측되며, 현재 파일 실행 시 args 로 --tokenizer_path 로
emji_tokenzer 를 지정하게 되면 내부의 model.json 을 불러들여 옵니다.
KoBARTConditionalGeneration 클래스 실행 시에, args 를 제공해주면 self.tokenizer 로 해당 json 을 지정하게 됩니다.

이후
get_pytorch_kobart_model(cachedir=".")
를 실행하면
1. kobart_base_cased_xxx...zip 이 생성됩니다.
    kobart tokenizer 와 동일하게, 입력 파라미터로 zip 저장하는 캐시 폴더를 입력받고, 현재 폴더에 저장합니다.
2. kobart_from_pretrained 라는 폴더가 생성됩니다.
해당 파일은 모델 압축 파일, kobart_from_pretrained 는 압축을 푼 원본 모델이라고 생각됩니다.
실제 모델은 kobart_from_pretrained/pytorch_model.bin 이라고 추측되며, 현재 파일 실행 시 args 로 --model_path 를 주게 되면, 
KoBARTConditionalGeneration 클래스 내부의 BartForConditionalGeneration 이라는 클래스를, 모델을 불러들여와 생성합니다.
따라서, KoBARTConditionalGeneration.model 이 실제 KoBART 모델이 됩니다.



이제, 이 파일을 실행시킬 때의 예제를 파악해 보겠습니다.
$ python kobart_chit_chat.py  
    --gradient_clip_val 1.0                 - 아직 파악 중입니다.
    --max_epochs 3                          - 최대 학습 epoch 수입니다. 기본 3으로 설정되어 있는 것 같습니다.
    --default_root_dir logs                 - 학습 시 일정한 체크포인트마다 로그를 생성하는데, 해당 로그의 저장 폴더를 지정합니다.
    --model_path kobart_from_pretrained     - 위에서 설명한, kobart 모델 경로입니다.
    --tokenizer_path emji_tokenizer         - 위에서 설명한, kobart tokenizer 경로입니다.
    --chat                                  - chat 을 주게 되면, 학습 완료 후 종료하지 않고 유저의 입력에 따른 출력을 반환합니다.
    --gpus 1                                - 실행할 때, 얼마나 많은 GPU 를 사용할 것인지입니다. 아쉽게도 koBART 모델은 GPU 기반으로 작동한다고 되어 있습니다.
                                              CPU 기반으로는 작동하지 않는 것으로 파악됩니다.

이제 실행하게 되면,
if __name__ == "__main__": 으로 가게 됩니다.
    1. Base 라는, 기본 pytorch lightning model 기반으로 한 클래스에서 parsing argument 를 추가합니다.
        - batch size (--batch_size)         학습 배치 사이즈   
        - learning_rate (--lr)              학습률
        - warmup_ratio (--warmup_ratio)     warmup ratio...? 파악이 필요해 보입니다.
        - model_path (--model_path)         koBART 모델 경로입니다.
    2. ArgBase 라는 기본 클래스 내부 함수가 parsing argument 를 추가합니다.
        - --train_file :        학습을 진행할 csv 파일 
        - --test_file :         학습을 테스트할 csv 파일
        - --tokenizer_path :    bart tokenizer 경로
        - --batch_size :        배치 사이즈 파라미터 (기본 14)
        - --max_seq_len :       토크나이저로 잘라냈을 때 입력할 수 있는 최대 토큰 
    3. ChatDataModule 라는 클래스 내부 함수가 parsing argument 를 추가합니다.
        - --num_workers :       데이터를 로딩할 때 사용할 워커 (아마 스레드) 수
        
    4. 이후, 받은 args 를 토대로 KoBARTConditionalGeneration 클래스의 인스턴스 생성
        1). 모델 경로를 토대로 모델 생성 (self.model)
        2). self.model.train() 으로 모델 학습 진행 (이미 사전학습되어있는 모델이기에 어떤 의미인지는 모름... 불분명한 호출)
        3). beginning_of_sentence, end_of_sentence 토큰 정의
        4). tokenizer 정의, tokenizer 경로 토대로 생성
            (1). 여기서 eos / bos 토큰을 정의합니다. 저희가 추가하고자 하는 감성에 해당하는 special token 주입이 가능할까요?
    
    5. ChatDataModule 클래스의 인스턴스를 생성합니다.
        1). 입력 파라미터 중 눈여겨봐야할 것은 tok_vocab 입니다. emji_tokenizer/model.json 이며, 해당 파일이 tokenizer 의 결과값으로 추정됩니다.
    
    6. ModelCheckpoint 를 init 합니다.
        1). 학습 시 각 모델의 체크포인트마다 어떤 값을 모니터하고, 어떻게 저장할 지 정의해 놓고 있습니다.
        
    7. tb_logger, lr_logger 도 동일하게 로그를 찍는 데 존재하는 모듈이라고 생각합니다.
    
    8. trainer.fit 을 이용해 모델을 학습시킵니다. 중요하게 봐야 할 부분입니다.
        1). 기본적으로, Base 모델은 pytorch lightning 의 LightningModule 을 사용했다는 것입니다. 
            https://baeseongsu.github.io/posts/pytorch-lightning-introduction/ 를 참고했습니다.
        2). KoBARTConditionalGeneration 은 총 2개의 step 에 대해 loop 을 설정했습니다.
            (1). training 시에 step 마다 실행하는 training_step
                - 각 batch 를 수행하고 loss 를 return 합니다.
            (2). validation 시에 step 마다 실행하는 vaildation_step
                - training 과 동일합니다.
            (3). forward 가 어디서 사용되는지
        3). 데이터는 dm (ChatDataModule) 에서 받아옵니다.
            (1). setup 에서 데이터를 지정하고 불러옵니다. 아마... 여기서 special token 을 지정해줘야하지 않을까요?
        4). optimizer 또한 dm.configure_optimizer 에서 진행합니다.
         
    9. args 에 --chat 을 넣었으면, 유저와 대화를 시도합니다.   
    
    고민해봐야 할 점
        BART 의 인코더 뒤에 FNN 을 붙여야하는데, 해당 부분이 koBART 에선 보이지 않습니다. 아마 모델을 그대로 불러오기만 해서 그런 것으로 생각됩니다.
        다른 BART 구현체도 찾아봐야할 것 같습니다 (huggingface 같은 데에서 검색 필요)
        
        만약 KoBART 를 사용한다면, special token 을 저희가 직접 만들어서 넣어주는 수밖엔 없어보입니다.
"""


parser = argparse.ArgumentParser(description='KoBART Chit-Chat')


parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='Chatbot_data/train.csv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='Chatbot_data/test.csv',
                            help='test file')

        parser.add_argument('--tokenizer_path',
                            type=str,
                            default='tokenizer',
                            help='tokenizer')
        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_seq_len',
                            type=int,
                            default=60,
                            help='max seq len')
        return parser


class ChatDataset(Dataset):
    def __init__(self, filepath, tok_vocab, max_seq_len=128) -> None:
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_vocab,
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

    def __len__(self):
        return len(self.data)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
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

    def __getitem__(self, index):
        record = self.data.iloc[index]
        q, a = record['Q'], record['A']
        # logger.warning(f"{q}|\t{a}|")
        # logger.warning(f"{self.tokenizer.tokenize(q)}|\t{self.tokenizer.tokenize(a)}|")
        q_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(a) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            q_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            a_tokens, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            a_tokens[1:(self.max_seq_len + 1)])
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100]
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}


class ChatDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok_vocab,
                 max_seq_len=128,
                 batch_size=32,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok_vocab = tok_vocab
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = ChatDataset(self.train_file_path,
                                 self.tok_vocab,
                                 self.max_seq_len)
        self.test = ChatDataset(self.test_file_path,
                                self.tok_vocab,
                                self.max_seq_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=False)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


class Base(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, **kwargs) -> None:
        super(Base, self).__init__()
        new_params = hparams.__dict__
        for key in new_params:
            self.hparams[key] = new_params[key]
        # self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        # hparams 는 args 임 (main.py init 시 입력 파라미터)
        print(hparams)
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        # model_path 에서 kobart_from_pretrained 모델 주입
        self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(self.hparams.tokenizer_path, 'model.json'),
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>',
        )

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'],
                          labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)


    def chat(self, text):
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        res_ids = self.model.generate(torch.tensor([input_ids]),
                                            max_length=self.hparams.max_seq_len,
                                            num_beams=5,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            bad_words_ids=[[self.tokenizer.unk_token_id]])        
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace('<s>', '').replace('</s>', '')


if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = ChatDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = KoBARTConditionalGeneration(args)

    dm = ChatDataModule(args.train_file,
                        args.test_file,
                        os.path.join(args.tokenizer_path, 'model.json'),
                        max_seq_len=args.max_seq_len,
                        num_workers=args.num_workers)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1,
                                                       prefix='kobart_chitchat')
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
    if args.chat:
        model.model.eval()
        while 1:
            q = input('user > ').strip()
            if q == 'quit':
                break
            print("Simsimi > {}".format(model.chat(q)))
