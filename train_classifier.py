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


def to_bitfit(model, verbose=False):
    """
    turn off anything except bias and classification head 
    in a transformer model
    """
    if verbose:
        print('most parameters will be turned off.')
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.lm_head.named_parameters():
        param.requires_grad = True
        
    return model

def main(args):
    data = pd.read_csv(args.data_path)

    data['input'] = data['sentence']

    le = preprocessing.LabelEncoder()
    data['labels'] = le.fit_transform(data['type'])

    joblib.dump(le, os.path.join(args.output_dir, 'label_encoder.joblib'))

    train_df, test_df = train_test_split(data, test_size=0.1, random_state=0)
    # train_df = pd.DataFrame({'source':list(train['input'].astype(str)), 'target':list(train['output'].astype(str))})
    train_dataset = Dataset.from_pandas(train_df)
    # test_df = pd.DataFrame({'source':list(test['input']), 'target':list(test['output'])})
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    def preprocess_function(examples):
        inputs = examples['input']
        model_inputs = tokenizer(text = inputs, max_length=148, padding='max_length', truncation=True)
        
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)

    label2id = dict(zip(le.classes_, le.transform(le.classes_)))
    id2label = {v: k for k, v in label2id.items()}
    print(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name_or_path,
        num_labels=len(label2id)
        )

    # if you really can't fit your model in try bitfit, which only adjusted bias terms
    # model = to_bitfit(model, verbose=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset, 
        eval_dataset=test_dataset,
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
        data_path='./data/political_stance/stance_detection.csv', 
        base_model_name_or_path="launch/POLITICS",
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
