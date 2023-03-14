"""
Some HF utils I used for some other proj
"""
import os
import math
import torch
import pandas as pd
import numpy as np

from dataclasses import dataclass
from transformers import BertModel, BertConfig
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput,EvalLoopOutput
from typing import Optional, Union
from datasets import Dataset
from copy import deepcopy
from sentence_transformers import SentenceTransformer

from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
import time
import datetime

from utils import (
    ReviewHistory, 
    DataReader,
    move_to_device
)

from sklearn.metrics import balanced_accuracy_score

class ProgressCallback(TrainerCallback):

    def setup(self, total_epochs, print_every=1): 
        self.total_epochs = total_epochs 
        self.current_epoch = 0
        self.epoch_start_time = None
        self.current_step = 1
        self.global_start_time = time.time()
        self.print_every=print_every
        return self

    def on_step_begin(self, args, state, control, **kwargs):
        
        avg_time_per_step = (time.time() - self.global_start_time)/max(state.global_step,1 )
        eta = avg_time_per_step * (state.max_steps-state.global_step) / 3600
        if self.current_step % self.print_every == 0:
            print(
                'epoch: ', 
                self.current_epoch,
                ', step ',
                self.current_step, 
                '/', 
                state.max_steps // self.total_epochs, 
                '||', 
                datetime.datetime.now(),
                '|| ETA(hrs): ',
                round(eta,2)
                )
        self.current_step += 1
        

    def on_epoch_begin(self, args, state, control, **kwargs):
        print('[ProgressCallback]: current epoch: ', self.current_epoch,' / ', self.total_epochs)
        self.current_epoch += 1
        self.current_step = 1
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        print('[ProgressCallback]: epoch', self.current_epoch,' / ', self.total_epochs, ' done')
        print("--- %s hours ---" % ((time.time() - self.epoch_start_time)/3600) )


class CustomTrainingArguments(TrainingArguments):
    """
    HF tainer has that forced multi-gpu usage that 
    really breaks slurm scheduling.
    Use this to turn that off.
    """

    @property
    def world_size(self):
        return 1
    
class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('[CustomTrainer] manually setting n_gpu to 1! huggingface_utils.py')
        self.args._n_gpu = 1
        self.local_rank = -1
        
    
    def prediction_step(self,model,inputs,prediction_loss_only, ignore_keys):
        with torch.no_grad():
            if torch.cuda.is_available():
                output = model(
                    **move_to_device(inputs, torch.device('cuda'))
                )
            else:
                output = model(**inputs)
        
        return output.loss.detach().cpu()

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        ):
            """
            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
            Works both with or without labels.
            """
            args = self.args
            model = self._wrap_model(self.model, training=False)
            self.callback_handler.eval_dataloader = dataloader
            model.eval()
            batch_size = dataloader.batch_size
            num_examples = self.num_examples(dataloader)
            print(f"***** Running evaluation loop *****")
            print(f"  Num examples = {num_examples}")
            print(f"  Batch size = {batch_size}")
            loss_host = []
            
            for step, inputs in enumerate(dataloader):
                loss = self.prediction_step(
                    model, 
                    inputs, 
                    prediction_loss_only, 
                    ignore_keys=ignore_keys
                )
                
                
                loss_host += loss.repeat(batch_size).tolist()
                

                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            
            loss_host = torch.tensor(loss_host)
            
            
                    
            metrics = {
                'eval_loss':torch.mean(loss_host).item()
            }

            print('[CustomTrainer]: Evaluation done)', metrics)

            output = EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_examples)
            return output




def prepare_dataset(data_path, index_dir, sentence_transformer_name='all-mpnet-base-v2', maybe_load_from=None):
    corpus = DataReader(
        data_path,
        index_dir
    )

    review_history = ReviewHistory(
        corpus.train, valid_data = corpus.valid, test_data = corpus.test
    ).build_embedded_text_table(
        SentenceTransformer(sentence_transformer_name), 
        torch.device('cuda'),
        maybe_load_from=maybe_load_from
    )

    # from prepare_evidence import prepare_evidence
    # prepare_evidence(corpus, review_history)

    train_dataframe = pd.DataFrame(corpus.train)
    valid_dataframe = pd.DataFrame(corpus.valid)

    train_dataset = Dataset.from_pandas(train_dataframe)
    valid_dataset = Dataset.from_pandas(valid_dataframe)
    
    
    return train_dataset, valid_dataset, review_history