
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
    📝 Train Method Notes 
    
    📶 Training Steps: During training, the model processes batches of data (training steps).

    📶 Evaluation Steps: After every eval_steps training steps, the model is evaluated on a validation dataset, and metrics (e.g., loss, accuracy) are computed. 
    This helps you monitor the model's performance during training.
    Evaluations can be linked to checkpoints, meaning the model will be saved after each evaluation step.

    """
        
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

    print("🚶 Size of eval_steps: ", eval_steps)
    print(f"📍 Result path:{args.result_path}")

    """
    📝 Notes on arguments: 
    save_total_limit -- means only the 15 most recent checkpoints will be kept on disk
    logging_steps -- controls how often training logs are printed — like loss, learning rate, etc.
        logging_dir='./logs',  # Optional, for TensorBoard
    """
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
                        logging_steps=100,
                                     
                    )
    
   
    trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), SaveTopModelsCallback(args.save_model_fold_path)]
 
    )

    print(f"📤 Supposed to be destination of saving models -> {args.save_model_fold_path}")
 
    print("😊 Before training 😊")
    trainer.train()
    print("😮‍💨 After training 😮‍💨")
    
    return model





"""
This SaveTopModelsCallback class is a custom callback for Hugging Face's Trainer that automatically:

1- Tracks the top-k best models during training (based on lowest validation loss)

2- Saves them to disk (as .pt files)


"""

# 🔧 Functionality: Makes a deep copy of the model's weights and biases using .clone() to avoid modifying original weights by accident.
# 🤔 Why? When saving the top-k models, we don't want to accidentally overwrite or corrupt the original model's parameters later.

def deep_copy_state_dict(state_dict):
    copy_dict = {}
    for key, value in state_dict.items():
        copy_dict[key] = value.clone()
    return copy_dict


class SaveTopModelsCallback(TrainerCallback):
    
    def __init__(self, save_path, top_k=2):
        # Directory where the model checkpoints will be saved -- ckpts_asap/t5-small/0
        self.save_path = save_path
        print(f"🚨 self.save_path is: {save_path} 🚨")
        # Number of top models to keep
        self.top_k = top_k
        # A list to store tuples like (loss, step, model_weights)
        self.top_models = []  

# This method is automatically called every time the trainer evaluates the model, e.g., every eval_steps
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):

        print("💥 Inside on_evaluate method -- SaveTopModelsCallback")
        
        # Get the current evaluation loss and training step
        current_loss = metrics['eval_loss']
        current_step = state.global_step

        # Temporarily move model to CPU for safe copying + Create a deep copy of the model's weights
        kwargs["model"] = kwargs["model"].cpu()
        model_state_dict = deep_copy_state_dict(kwargs['model'].state_dict())  
        kwargs["model"] = kwargs["model"].to(args.device)

        # Add the model to self.top_models list with its loss, step, and weights
        self.top_models.append((current_loss, current_step, model_state_dict))
        
        # Sort the list based on loss (lowest first) and keeps only the top k best models
        self.top_models.sort(key=lambda x: x[0])  
        self.top_models = self.top_models[:self.top_k]  

        # Call helper method to save the top models and delete old ones.
        self.cleanup_and_save_top_models()

# This method removes previously saved checkpoints and writes the current top models to disk
    def cleanup_and_save_top_models(self):

        print("💥 Inside cleanup_and_save_top_models method -- SaveTopModelsCallback")
        print(f"🚨File Search in: {self.save_path}🚨")
        # Loops through all files in the save_path directory
        for filename in os.listdir(self.save_path):
            if filename.startswith("checkpoint"):
                print(f"💥💥💥💥💥💥💥💥💥 Found file that starts with ckpt:{filename} -- SaveTopModelsCallback 💥💥💥💥💥💥💥💥💥💥💥")
                os.remove(os.path.join(self.save_path, filename))
        
    #   Loop through each top model (after sorting). rank will be 0 for the best, 1 for the second-best, etc.
        for rank, (loss, step, state_dict) in enumerate(self.top_models):
            print("💥💥💥💥💥💥💥 Inside rank loop -- SaveTopModelsCallback 💥💥💥💥💥")
            model_path = os.path.join(self.save_path, f"checkpoint-{rank+1}-loss-{loss:.4f}")
            th.save(state_dict, model_path)
            print(f"🔴🟠🟡🟢🔵🟣 Saved top {rank+1} model to {model_path} with loss {loss:.4f}🔴🟠🟡🟢🔵🟣")

