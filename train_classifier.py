import pandas as pd
from datasets import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"]="./huggingface_cache"
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_utils import CustomTrainer, Trainer, CustomTrainingArguments
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback, TrainerCallback
from huggingface_utils import ProgressCallback
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib 
import os


def main(args):
    from datasets import load_dataset

    dataset = load_dataset("sst2")

    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    def preprocess_function(examples):
        inputs = examples['sentence']
        model_inputs = tokenizer(text = inputs, max_length=148, padding='max_length', truncation=True)
        
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function)
    val_dataset = val_dataset.map(preprocess_function)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name_or_path,
        num_labels= len(set(dataset['train']['label']))
    )


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=CustomTrainingArguments(
            load_best_model_at_end = True,
            output_dir = args.output_dir,
            save_strategy = 'epoch',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            save_total_limit =1,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            logging_steps=1
        )
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = args.tolerance_steps,
        early_stopping_threshold=1e-7
    )
    trainer.add_callback(early_stopping_callback)
    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generator Model')
    
    from easydict import EasyDict as edict
    args = dict(
        data_path='', 
        base_model_name_or_path="bert-base-uncased",
        output_dir='./checkpoints',
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        tolerance_steps=3,
        learning_rate=3e-5
    )
    args = edict(args)
        
    
    main(args)
