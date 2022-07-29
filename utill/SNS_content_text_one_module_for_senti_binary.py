import os	
import pandas as pd	
import numpy as np
from pytorch_lightning import plugins
from pytorch_lightning.utilities import distributed
from sklearn.utils.multiclass import check_classification_targets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertModel
import torch	
from torch.utils.data import Dataset, DataLoader, TensorDataset	
from torch.optim.lr_scheduler import ExponentialLR	
from pytorch_lightning import LightningModule, Trainer
import re	
import emoji	
from soynlp.normalizer import repeat_normalize
import time	
import unicodedata
from multiprocessing import Pool
import warnings
from collections import OrderedDict
# from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torch.nn import functional as F
import torch.nn as nn
warnings.filterwarnings("ignore")

class Model(LightningModule):
    
    def __init__(self, options):
        super().__init__()
        self.args = options
        # self.bert = AutoModelForSequenceClassification.from_pretrained(self.args.pretrained_model, num_labels=2)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.args.pretrained_tokenizer if self.args.pretrained_tokenizer else self.args.pretrained_model
        #

        try:
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join('..','model','KcELECTRA-base'),
                                                                      local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join('..','model','KcELECTRA-base'), local_files_only=True)
        except:
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(os.getcwd(),'model','KcELECTRA-base'),local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.getcwd(),'model','KcELECTRA-base'),local_files_only=True)

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    # 이때의 테스트 스텝은 각 배치별로 도는 것임
    def predict_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data)
        logits = output.logits
        _predicts = logits.argmax(dim=-1)
        prob = torch.sigmoid(logits).max(dim=-1)[0]
        # prob = F.softmax(logits).max(dim=-1)[0]
        
        output = OrderedDict({
            'prob': prob.detach().cpu().numpy().tolist(),
            'y_pred': _predicts.detach().cpu().numpy().tolist(),
            "labels": labels.detach().cpu().numpy().tolist()
            })
        return output

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')

        if self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        # lr_scheduler 는 exp 로 고정인데 코드상에서 분기가 있는 이유는?
        # elif self.args.lr_scheduler == 'cos':
        #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

class SnsContentClassifier:
    def __init__(self):
        self.args = None
        self.content_df = None


    def load_model(self):
        # 21 categories
        # checkpoint_path = '../bert_model/epochepoch=2-val_accval_acc=0.9997.ckpt'    
        # checkpoint_path = './lightning_logs/version_30677/checkpoints/epochepoch=1-val_accval_acc=0.7574.ckpt'
        # checkpoint_path = "./lightning_logs/version_30707/checkpoints/epochepoch=1-val_accval_acc=0.4173.ckpt"
        # checkpoint_path = "./lightning_logs/version_30764/checkpoints/epochepoch=4-val_accval_acc=0.9215.ckpt"
        # checkpoint_path = "./lightning_logs/version_30784/checkpoints/epochepoch=11-val_accval_acc=0.9430.ckpt"
        try:
            pretrained_model = Model.load_from_checkpoint(checkpoint_path=os.path.join(os.getcwd(),
                                            'model','sentiment_pytorch','epochepoch=4-val_accval_acc=0.9215.ckpt'), options=self.args)
        except OSError:
            pretrained_model = Model.load_from_checkpoint(
                checkpoint_path=os.path.join('..', 'model', 'sentiment_pytorch',
                                             'epochepoch=4-val_accval_acc=0.9215.ckpt'), options=self.args)
        pretrained_model = pretrained_model.to("cuda")
        pretrained_model.eval()
        # pretrained_model.freeze()

        return pretrained_model

    # multiprocess split with cpu_count of num cores
    @staticmethod
    def parallelize_dataframe(df, func):
        num_cores = os.cpu_count()
        if len(df) < num_cores:
            num_cores = len(df)
        num_cores = 1
        # add van : 병렬 처리를 위해 df를 cpu_core 수로 분할
        df_split = np.array_split(df, num_cores)
        # cpu 병렬 처리를 위한 multiprocessing.Pool 생성
        pool = Pool(num_cores)
        # df_split 데이터를 func으로 병렬 처리하고, 처리결과를 pd.concat으로 합침, 즉 multiprocessing_func_preprocess() 병렬 실행
        concatenated = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return concatenated

    @staticmethod
    def multiprocessing_func_preprocess(df):
        # preprocess functions
        # add van : 정규표현식 패턴 Oject 생성
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        url_pattern2 = re.compile(r'http[s]?\s?(com|kr)?\w*')
        url_pattern3 = re.compile(r'w{3}?\s?(com|kr)?\w*')
        phone_pattern= re.compile(r'010\s?[\w|\d]{4}\s?[\w|\d]{4}')
        email_pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        num = re.compile(r'([일|이|삼|사|오|육|칠|팔|구|십]*\s?[십|백|천|만|억]+\s?)+')
        # tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')
        try:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join('..','model','KcELECTRA-base'), local_files_only=True)
        except:

            tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.getcwd(),'model','KcELECTRA-base'),local_files_only=True)

        # tokenizer = BertTokenizer.from_pretrained('../pretrained_tensorflow_model',local_files_only=True)
        # add van : 위에 정의한 정규표현식 Object를 사용하여 패턴이 매치되는 문자열을 치환함, sub() 사용
        def clean(x):
            x = pattern.sub(' ', x)   # add van : x에 특수문자가 있으면 빈 문자열로 치환
            x = url_pattern.sub('', x)
            x = x.strip('\n')
            x = email_pattern.sub('',x)
            x = phone_pattern.sub('번호',x)
            x = url_pattern2.sub(' ', x)
            x = url_pattern3.sub(' ', x)
            x = num.sub('숫자',x)
            x = unicodedata.normalize('NFC',x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x
        # df['document'] = df['title'].astype(str) + ' ' + df['content'].astype(str)
        # df.rename(columns={"content": "document"}, inplace=True)
        
        # map function for each row
        # add van :  df['document']를 위에 정의된 패턴(문자, 숫자, email, phone 등)으로 전처리(clean) 함
        df['document'] = df['document'].map(
            lambda x: tokenizer.encode(clean(str(x)), padding='max_length', max_length= 300, truncation=True)
        )

        return df

    def preprocess_dataframe(self, df):
        pre_time = time.time()
        _df = self.parallelize_dataframe(df, self.multiprocessing_func_preprocess)  # multiprocess with cpu
        print(f'Elapsed for preprocess data: {round(time.time() - pre_time, 3)} seconds')
        return _df

    def test_dataloader(self):
        load_time = time.time()
        _df = self.preprocess_dataframe(self.content_df)

        # create label column w/o any value for testing
        # _df['label'] = 0
        dataset = TensorDataset(
            torch.tensor(_df['document'].to_list(), dtype=torch.long),
            torch.tensor([i for i in range(len(_df['document'].to_list()))], dtype=torch.long),
        )
        print(f'Elapsed for data loading(read+preprocess): {round(time.time() - load_time, 3)} seconds')
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers
        )

    # TRAINING THE TRAINER FUNCTION
    def training(self, pretrained_model):
        train_start = time.time()
        # Trainer
        trainer = Trainer(
            max_epochs=self.args.epochs,
            fast_dev_run=self.args.test_mode,
            num_sanity_val_steps=None if self.args.test_mode else 0,
            auto_scale_batch_size=self.args.auto_batch_size if self.args.auto_batch_size and not self.args.batch_size else False,
            # For GPU Setup
            deterministic=torch.cuda.is_available(),
            # gpus=-1,
            gpus =1 ,
            accelerator='dp' if torch.cuda.is_available() else None,
            # distributed_backend='ddp',
            # plugins=DDPPlugin(),
            # num_nodes=1,
        
            # gpus=[2,3],
            precision=16 if self.args.fp16 else 32,
        )

        # train_result = trainer.test(pretrained_model, test_dataloaders=self.test_dataloader(),verbose=False)
        outputs = trainer.predict(pretrained_model,dataloaders=self.test_dataloader())
        # train_result = trainer.predict(pretrained_model, dataloaders=self.test_dataloader())
        print(f'Elapsed for KcBert to test: {round(time.time() - train_start, 3)} seconds')
        # return train_result
        
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        #     gather = [None] * torch.distributed.get_world_size()
        #     torch.distributed.all_gather_object(gather, train_result)
        #     # outputs = sum([x for xs in gather for x in xs],[])
        #     outputs = [x for xs in gather for x in xs]
        prob = [i.detach().cpu().numpy().tolist()[0] for i in sum([k['prob'] for k in outputs],[])]
        y_pred = [i.detach().cpu().numpy().tolist()[0] for i in sum([k['y_pred'] for k in outputs],[])]
        orders = [i.detach().cpu().numpy().tolist()[0] for i in sum([z['labels'] for z in outputs],[])]
        tmp_dic = OrderedDict({ name:value for name, value in zip(orders, y_pred) })
        tmp_dic2 = OrderedDict({ name:value for name, value in zip(orders, prob) })
        sorted_dic = sorted(tmp_dic.items()) ##순서를 다시 맞춰줌
        sorted_dic2 = sorted(tmp_dic2.items())
        result = [value for key, value in sorted_dic]
        result2 = [value for key, value in sorted_dic2]
        return result, result2    # result: 예측값

    def true_y(self, x):
        if x == 0:
            return 'neg'
        elif x == 1:
            return 'pos'
        # elif x == 2:
        #     return '부동산'
        # elif x == 3:
        #     return '수리'
        # elif x == 4:
        #     return '이벤트'
        # elif x == 5:
        #     return '인사글'
        # elif x == 6:
        #     return '종교'
        # elif x == 7:
        #     return '주식'
        else:
            return 'neu'
    
    def mapping_process(self, content_df, train_result, train_result2):
        start_time = time.time()
        # content_df['y_hat'] = train_result[0]['y_pred']
        content_df['prob'] = train_result2
        content_df['y_hat'] = train_result
        content_df['y_hat_label'] = content_df['y_hat'].map(lambda x: self.true_y(x))

        print(f'Time taken for mapping_process {round(time.time() - start_time, 3)} seconds')

        return content_df  # 모델 예측값 + 룰베이스 예측값: 최종결과값
    def run(self, content_df):
        class Arg:
            random_seed: int = 42
            # Transformers PLM name
            pretrained_model: str = 'beomi/KcELECTRA-base'
            # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
            pretrained_tokenizer: str = ''
            # Let PyTorch Lightening find the best batch size
            auto_batch_size: str = 'power'
            # Optional, Train/Eval Batch Size. Overrides `auto_batch_size`
            batch_size: int = 256
            # Starting Learning Rate
            lr: float = 5e-6
            # Max Epochs
            epochs: int = 20
            # Max Length input size
            max_length: int = 300
            # Report (Train Metrics) Cycle
            report_cycle: int = 100
            # Multi cpu workers
            # cpu_workers: int = os.cpu_count()
            cpu_workers = 1
            # KcBERT_Garbage2.0 Mode enables `fast_dev_run`
            test_mode: bool = False
            optimizer: str = 'AdamW'
            lr_scheduler: str = 'exp'
            # Enable train on FP16
            fp16: bool = True
            # Enable TPU with 1 core or 8 cores
            tpu_cores: int = 0

        self.args = Arg()
        self.content_df = content_df

        pretrained_model = self.load_model()

        train_result, train_result2 = self.training(pretrained_model)

        result_df = self.mapping_process(content_df=content_df,train_result=train_result, train_result2=train_result2)

        return result_df
        # return train_result

