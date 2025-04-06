
import os
import random
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import TrainerCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
import torch as th
from tqdm import tqdm
from evaluation import quadratic_weighted_kappa
import warnings
import traceback

warnings.filterwarnings("ignore")

def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if th.cuda.is_available():
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def read_data(data_path):

    print("utils: read data")
    
    df = pd.read_csv(data_path)
    
    dataset = Dataset.from_pandas(df)
    
    return dataset

def preprocess_data(examples, tokenizer,args):

    print("utils:preprocess_data")

    
    essay = tokenizer([ "<essay> "+ example for example in examples["t5_input"]], max_length=512, truncation=True, padding="max_length")
    
            
    if args.llm == "gpt":
        criteria = tokenizer([ " <rationale> "+ example for example in examples["gpt_criteria"]], max_length=512, truncation=True, padding="max_length")
    else:
        criteria = tokenizer([ " <rationale> "+ example for example in examples["llama_criteria"]], max_length=512, truncation=True, padding="max_length")

    essay["input_ids"] = [sublist1 + sublist2 for sublist1, sublist2 in zip(essay["input_ids"],criteria["input_ids"])]
    essay["attention_mask"] = [sublist1 + sublist2 for sublist1, sublist2 in zip(essay["attention_mask"],criteria["attention_mask"])]

    with tokenizer.as_target_tokenizer():
        
        labels = examples["t5_output"]
        
        if args.data == "asap":
            if "t5" in args.model_name:
                labels = tokenizer(labels, max_length=64, truncation=True, padding="max_length")
            else:
                labels = tokenizer(labels, max_length=256, truncation=True, padding="max_length")
        else:
            if "flan-t5-base" in args.model_name:
                labels = tokenizer(labels, max_length=256, truncation=True, padding="max_length")
            else:
                labels = tokenizer(labels, max_length=64, truncation=True, padding="max_length")
        
        
        

    essay["labels"] = labels["input_ids"]
    
    return essay




