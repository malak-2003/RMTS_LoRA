
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
from torch.nn import functional as F

import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


warnings.filterwarnings("ignore")


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    if th.cuda.is_available():
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def read_data(data_path):
    
    df = pd.read_csv(data_path)
    
    dataset = Dataset.from_pandas(df)
    
    return dataset

def preprocess_data(examples, tokenizer,args):
    
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

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    print("🧠 Trainable Parameters:")
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            # print(f"  ✅ {name}: {param.numel():,}")
        # else:
            # print(f"  ❌ {name}: frozen")

    print(f"\n🔢 Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,}")
    print(f"📉 Percentage trainable: {100 * trainable / total:.4f}%")



def train(model, tokenizer, train_dataset, dev_dataset, args=None):

    print("😊 Begining of training!!!!!!!!! 😊")
   

    if args.data == "asap":
        eval_steps = int(np.ceil(5000/(args.train_batch_size/4)))
     
    else:
        eval_steps = 1600

    print("🚶 Size of eval_steps: ", eval_steps)
    print(f"📍 Result path:{args.result_path}")


    output_dir_path=f"./{args.result_path}"
    
   
    training_args = Seq2SeqTrainingArguments(
                        output_dir=output_dir_path,           
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


def asap_test(tokenizer, model, test_data, args):
    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
    accuracy_result = dict()
    precision_result = dict()
    recall_result = dict()
    f1_result = dict()


    trait_map = {
    1: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    2: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    3: ["overall", "content", "prompt adherence", "language", "narrativity"],
    4: ["overall", "content", "prompt adherence", "language", "narrativity"],
    5: ["overall", "content", "prompt adherence", "language", "narrativity"],
    6: ["overall", "content", "prompt adherence", "language", "narrativity"],
    7: ["overall", "content", "organization", "style", "conventions"],
    8: ["overall", "content", "organization", "voice", "word choice", "sentence fluency", "conventions"]
    }

    compound_keys = {
    'sentence fluency': 'sentence-fluency',
    'word choice': 'word-choice',
    'prompt adherence': 'prompt-adherence'
    }

    for p in range(1,9):
        if p==0:
            print("🗂️ Initializes dictionaries to store predictions")

        pred_dic[p] = dict()
        true_dic[p] = dict()
        qwk_result[p] = dict()
        trait_list = trait_map[p]
        for trait in trait_list:
            pred_dic[p][trait] = list()
            true_dic[p][trait] = list()
            qwk_result[p][trait] = 0.0
            accuracy_result[p][trait] = 0.0
            precision_result[p][trait] = 0.0
            recall_result[p][trait] = 0.0
            f1_result[p][trait] = 0.0

    model.eval()
    batch_size = 128
    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i+batch_size]
            input_ids_all  = th.tensor(test['input_ids']).to(args.device)
            attention_mask =  th.tensor(test['attention_mask']).to(args.device)
            
            essay_input_ids = input_ids_all[:,:512]
            essay_attention_mask = th.ones(essay_input_ids.size(), dtype=th.long).to(model.device)
            encoder_outputs = model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            
            criteria_ids = input_ids_all[:,512:]
            criteria_attention_mask = th.ones(criteria_ids.size(), dtype=th.long).to(model.device)
            criteria_encoder_outputs = model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
            encoder_outputs.last_hidden_state = model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1)

            labels = test['t5_output']
            prompts = test["essay_set"]

            decoder_start_token_id = model.config.decoder_start_token_id
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            
            outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 64, num_beams =1)
            
            flag= True

            for i, (output, true) in enumerate(zip(outputs, labels)):
                
                pred = tokenizer.decode(output, skip_special_tokens=True)
                try:
                    pred_text = pred
                    for key, replacement in compound_keys.items():
                        pred_text = pred_text.replace(key, replacement)
                    items = pred_text.split(', ')                    

                    pred_result = {}
                    
                    for item in items:

                        key, value = item.split(' ', 1)
                        key = key.replace('-', ' ') 
                        if value == 'nan':
                            value = np.nan
                        else:
                            value = int(value)
                        pred_result[key] = value
                    
 
                    true_text = true
                    for key, replacement in compound_keys.items():
                        true_text = true_text.replace(key, replacement)
                    items = true_text.split(', ')
                    true_result = {}
                    for item in items:
                        key, value = item.split(' ', 1)
                        key = key.replace('-', ' ')  
                        if value == 'nan':
                            value = np.nan
                        else:
                            value = int(value)
                            true_result[key] = value

                    prompt = prompts[i]
                
                    trait_list = trait_map[prompt]

                    if flag:
                        print(f"🆘🆘 Pred_result : {pred_result}")
                        print(f"✅✅ True_result : {true_result}")
                        flag=False
                    
                    for trait in trait_list:
                        if trait not in pred_result:
                            print(f"⚠️ Trait '{trait}' not found in prediction for prompt {prompt}")
                            
                        if trait not in pred_result or trait not in true_result:
                            print(f"⚠️ Trait '{trait}' missing in prediction or true result for prompt {prompt}")
                            
                        if np.isnan(pred_result[trait]):
                            pred_dic[prompt][trait].append(0)
                            true_dic[prompt][trait].append(true_result[trait])
                            break
                        pred_dic[prompt][trait].append(pred_result[trait])
                        true_dic[prompt][trait].append(true_result[trait])
                    
                except Exception as e:
                    
                    print(f"An error occurred: {e}")
                    traceback.print_exc() 

                    continue
                   

        first = True
        for prompt in range(1,9):
            if first:
                print(f"🧮 Calculating QWK for Prompt {prompt}...")
                first = False


            trait_list = trait_map[prompt]
            
          
            for trait in trait_list:
                qwk_result[prompt][trait] = quadratic_weighted_kappa(np.array(pred_dic[prompt][trait]), np.array(true_dic[prompt][trait]))
               
                accuracy_result[prompt][trait] = accuracy_score(true_dic[prompt][trait], pred_dic[prompt][trait])
                
                precision_result[prompt][trait] = precision_score(true_dic[prompt][trait], pred_dic[prompt][trait], average='weighted', zero_division=1)
                
                recall_result[prompt][trait] = recall_score(true_dic[prompt][trait], pred_dic[prompt][trait], average='weighted', zero_division=1)
            
                f1_result[prompt][trait] = f1_score(true_dic[prompt][trait], pred_dic[prompt][trait], average='weighted', zero_division=1)
                                           
        log = "Test Result"

        for prompt in range(1,9):
            log += f"\n\n| Prompt: {prompt} |"
            log += f"\n| QWK: {qwk_result[prompt]} |"
            log += f"\n| Accuracy: {accuracy_result[prompt]} |"
            log += f"\n| Precision: {precision_result[prompt]} |"
            log += f"\n| Recall: {recall_result[prompt]} |"
            log += f"\n| F1-Score: {f1_result[prompt]} |"
        print(log)

        

    return qwk_result, pred_dic, true_dic


def deep_copy_state_dict(state_dict):
    copy_dict = {}
    for key, value in state_dict.items():
        copy_dict[key] = value.clone()
    return copy_dict

class SaveTopModelsCallback(TrainerCallback):
    
    def __init__(self, save_path, top_k=2):
        self.save_path = save_path
        print(f"🚨 self.save_path is: {save_path} 🚨")
        self.top_k = top_k
        self.top_models = []  


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):

        print("💥 Inside on_evaluate method -- SaveTopModelsCallback")
        
        current_loss = metrics['eval_loss']
        current_step = state.global_step

    
        kwargs["model"] = kwargs["model"].cpu()
        model_state_dict = deep_copy_state_dict(kwargs['model'].state_dict())  
        kwargs["model"] = kwargs["model"].to(args.device)

       
        self.top_models.append((current_loss, current_step, model_state_dict))
        
        
        self.top_models.sort(key=lambda x: x[0])  
        self.top_models = self.top_models[:self.top_k]  

        
        self.cleanup_and_save_top_models()


    def cleanup_and_save_top_models(self):

        print("💥 Inside cleanup_and_save_top_models method -- SaveTopModelsCallback")
        print(f"🚨File Search in: {self.save_path}🚨")
       
        for filename in os.listdir(self.save_path):
            if filename.startswith("checkpoint"):
                print(f"💥💥💥💥💥💥💥💥💥 Found file that starts with ckpt:{filename} -- SaveTopModelsCallback 💥💥💥💥💥💥💥💥💥💥💥")
                os.remove(os.path.join(self.save_path, filename))
        
        for rank, (loss, step, state_dict) in enumerate(self.top_models):
            print("💥💥💥💥💥💥💥 Inside rank loop -- SaveTopModelsCallback 💥💥💥💥💥")
            model_path = os.path.join(self.save_path, f"checkpoint-{rank+1}-loss-{loss:.4f}")
            th.save(state_dict, model_path)
            print(f"💾 Saved top {rank+1} model to {model_path} with loss {loss:.4f}")

def extract_traits(text):
    
    matches = re.findall(r'(\w+)\s+([\d.]+)', text)
    
    
    traits_dict = {trait: float(value.rstrip('.')) for trait, value in matches[:7]} 
    return traits_dict




