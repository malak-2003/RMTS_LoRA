
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
from peft import get_peft_model, LoraConfig

warnings.filterwarnings("ignore")

# NOT Tested 🕵🏻🆘
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

# Tested 🕵🏻✅
def read_data(data_path):

    print("👀 Read data -- utils")
    
    df = pd.read_csv(data_path)
    
    dataset = Dataset.from_pandas(df)
    
    return dataset

# Tested 🕵🏻✅
def preprocess_data(examples, tokenizer,args):

    print("🔄 Preprocessing Data -- utils")

    
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

# 🔄 Testing in process
def train(model, tokenizer, train_dataset, dev_dataset, args=None):
    """
    🛠️ Fine-Tuning Part of the Code 

    # LoRA configuration (add low-rank matrices to attention layers)
    lora_config = LoraConfig(
    r=8,  # Rank of the low-rank decomposition
    lora_alpha=32,  # Scaling factor for LoRA
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    task_type="SEQ_2_SEQ_LM"
    )

    # Apply LoRA to the model -- Way 1 (ChatGPT) 
    model = get_peft_model(model, lora_config)

    # Apply LoRA to the model -- Way 2 (Github)
    # Link 🔗:(https://gitlab.com/CeADARIreland_Public/llm-resources/-/blob/main/fine_tuning_template_script.py?ref_type=heads) 
    peft_config=lora_config,  -- put line inside trainer

    """



    if args.data == "asap":
        eval_steps = int(np.ceil(5000/(args.train_batch_size/4)))
        
    else:
        eval_steps = 1600

    """
    Training Steps: During training, the model processes batches of data (training steps).

    Evaluation Steps: After every eval_steps training steps, the model is evaluated on a validation dataset, and metrics (e.g., loss, accuracy) are computed. 
    This helps you monitor the model's performance during training.
    Evaluations can be linked to checkpoints, meaning the model will be saved after each evaluation step.

    """
        
    print("🚶 Size of eval_steps: ", eval_steps)
    print(f"📍 Result path:{args.result_path}")

    

    # These are the settings that control the training behavior. 
    training_args = Seq2SeqTrainingArguments(
                        output_dir=f"./{args.result_path}",           
                        evaluation_strategy="steps",      
                        eval_steps=eval_steps,                
                        per_device_train_batch_size=args.train_batch_size,    
                        per_device_eval_batch_size=args.train_batch_size,     
                        num_train_epochs=args.train_epochs,             
                        predict_with_generate=True,       
                        load_best_model_at_end=True,      
                        metric_for_best_model="loss",     
                        greater_is_better=False,          
                        save_steps=eval_steps,                 
                        save_total_limit=15,          
                        save_safetensors = False,
                        learning_rate=args.learning_rate,               
                    )
    
   
    trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), SaveTopModelsCallback(args.save_model_fold_path)]
 
    )
 
    print("😊 Before training 😊")
    trainer.train()
    print("😮‍💨 After training 😮‍💨")
    
    return model




