

import os
import random
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import TrainerCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import torch as th
from tqdm import tqdm
from evaluation import quadratic_weighted_kappa
import warnings
import traceback
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    Useful when using PEFT methods like LoRA to confirm only adapter layers are trainable.
    """
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
    print(f"🔧 Trainable parameters: {trainable:,}")
    print(f"📦 Total parameters: {total:,}")
    print(f"✅ Percentage trainable: {100 * trainable / total:.4f}%")


# Tested 🕵🏻✅
def train(model, tokenizer, train_dataset, dev_dataset, args=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",  # Normal Float 4 quantization
        bnb_4bit_use_double_quant=True,  # Nested quantization for better memory efficiency
        bnb_4bit_compute_dtype=th.bfloat16  # Computation dtype
    )
    
    # Apply QLoRA to the model
    model = prepare_model_for_kbit_training(model)
    

 # LoRA Configuration - Optimized for T5 architecture
    lora_config = LoraConfig(
        r=16,  # Slightly higher rank for generation tasks
        lora_alpha=32,
        target_modules=["q", "v", "wi", "wo"],  # T5-specific modules
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"  # Specific for T5 sequence-to-sequence
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show which parameters are trainable



    if args.data == "asap":
        eval_steps = int(np.ceil(5000/(args.train_batch_size/4)))
        # eval_steps= 500
        
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

    # output_dir_path= ""
    # if args.online_run:
    #     output_dir_path="/content/drive/MyDrive/QLoRA_Checkpoints"
    # else:
        
    output_dir_path=f"./{args.result_path}"
    
    print("📁 Should save in drive")
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
                        save_safetensors = False,
                        learning_rate=2e-4,  
                        logging_steps=100,
                        fp16=True,
                        optim="paged_adamw_8bit",         # 8-bit AdamW optimizer
                        gradient_accumulation_steps=4,     # MANUAL: Accumulate gradients (adjust based on GPU memory)
                        warmup_steps=100,
                        label_smoothing_factor=0.1
                        
                                     
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


# 🔄 Testing in process
from torch.nn import functional as F
def asap_test(tokenizer, model, test_data, args):

#Input Parameters
#tokenizer – Tokenizer object for converting input text into token IDs.
#model – Fine-tuned transformer model (T5, BART, Pegasus, or LED).
#test_data – Test dataset containing essay inputs and ground-truth scores.
#args – Configuration parameters (e.g., device type, model name).

#Output
#qwk_result – Dictionary storing QWK scores for each trait of each essay prompt.
#pred_dic – Dictionary storing predicted trait scores for each prompt.
#true_dic – Dictionary storing ground-truth trait scores.
    
#    These dictionaries will store predictions (pred_dic), actual scores (true_dic), and QWK results (qwk_result) for each essay prompt.
    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
#    Each prompt (1-8) has specific writing traits to be evaluated.
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
#    Handles compound trait names by replacing spaces with hyphens
    compound_keys = {
    'sentence fluency': 'sentence-fluency',
    'word choice': 'word-choice',
    'prompt adherence': 'prompt-adherence'
    }
#    Initializes dictionaries to store predictions, true labels, and QWK scores for each trait in each prompt.
    for p in range(1,9):
        print("🗂️ Initializes dictionaries to store predictions")
        pred_dic[p] = dict()
        true_dic[p] = dict()
        qwk_result[p] = dict()
        trait_list = trait_map[p]
        for trait in trait_list:
            pred_dic[p][trait] = list()
            true_dic[p][trait] = list()
            qwk_result[p][trait] = 0.0

# The model is set to evaluation mode
    model.eval()
#   Uses torch.no_grad() to disable gradient computation (since we're testing).Loops through the test dataset in batches of 128 essays.
    batch_size = 128
    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i+batch_size]
#        Loads the tokenized essays (input_ids_all) and attention masks.
            input_ids_all  = th.tensor(test['input_ids']).to(args.device)
            attention_mask =  th.tensor(test['attention_mask']).to(args.device)
#        Extracts the first 512 tokens of each essay (max input length for some models).
            essay_input_ids = input_ids_all[:,:512]
            essay_attention_mask = th.ones(essay_input_ids.size(), dtype=th.long).to(model.device)

#        Calls the encoder of the chosen model to process the essay.
            # print("🟡 Calls Encoder 🟡")
            encoder_outputs = model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)

                       

#           Extracts trait-related information (criteria) from the input.
            criteria_ids = input_ids_all[:,512:]
            criteria_attention_mask = th.ones(criteria_ids.size(), dtype=th.long).to(model.device)
            
#           Similar to essays, criteria are encoded using the model's encoder.
            criteria_encoder_outputs = model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
#           Combines essay encoding and criteria encoding
            encoder_outputs.last_hidden_state = model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1)

          
            labels = test['t5_output']
            prompts = test["essay_set"]

#   The decoder start token is used as input. The model generates predictions (trait scores) based on the essay.
            decoder_start_token_id = model.config.decoder_start_token_id
            
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            
            # hena kan fe if else w el eslse other model max_new_token was 256 -- make sure of number 
            outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 64, num_beams =1)
            
            flag= True

            for i, (output, true) in enumerate(zip(outputs, labels)):
                # Converts tokenized output back to readable text.
                pred = tokenizer.decode(output, skip_special_tokens=True)

                
                
                try:
                    

                    #Parses the predicted scores into a dictionary.
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
#               Stores the predictions and true scores for each trait.
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

                    # continue
                    break

                    
#   Computes Quadratic Weighted Kappa (QWK), which measures agreement between predictions and actual scores.
        for prompt in range(1,9):
            if prompt == 1:
                print("🏗️ Starting prompt extraction...")
            trait_list = trait_map[prompt]
            
            first = True
            for trait in trait_list:
                if first:
                    print(f"🧮 Calculating QWK for Prompt {prompt}...")
                    first = False
                qwk_result[prompt][trait] = quadratic_weighted_kappa(np.array(pred_dic[prompt][trait]), np.array(true_dic[prompt][trait]))
                                           
        log = "Test Result"
        for prompt in range(1,9):
            log += f"\n\n| Prompt: {prompt} |"
            log += f"\n| {qwk_result[prompt]} |"
        print(log)

        

    return qwk_result, pred_dic, true_dic


def feedback_test(tokenizer, model, test_data, args):

    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
    trait_list = ["conventions", "grammar", "phraseology", "vocabulary", "syntax", "cohesion"]

   
    for trait in trait_list:
        pred_dic[trait] = list()
        true_dic[trait] = list()
        qwk_result[trait] = 0.0


    model.eval()
    batch_size = 128
    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i+batch_size]
            input_ids_all  = th.tensor(test['input_ids']).to(args.device)
            attention_mask =  th.tensor(test['attention_mask']).to(args.device)

            essay_input_ids = input_ids_all[:,:512]
            essay_attention_mask = th.ones(essay_input_ids.size(), dtype=th.long).to(model.device)

            
            if 'bart' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            elif 't5' in args.model_name:
                encoder_outputs = model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            elif 'pegasus' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            elif 'led' in args.model_name:
                encoder_outputs = model.led.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            
           
            criteria_ids = input_ids_all[:,512:]
            criteria_attention_mask = th.ones(criteria_ids.size(), dtype=th.long).to(model.device)
            if 'bart' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                
            elif 't5' in args.model_name:
                criteria_encoder_outputs = model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                
                
            elif 'pegasus' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                
            elif 'led' in args.model_name:
                criteria_encoder_outputs = model.led.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.led.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                    
                
            labels = test['t5_output']

            
            decoder_start_token_id = model.config.decoder_start_token_id
            
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            
            if "flan-t5-base" in args.model_name:
                outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 256, num_beams =1)
            else:
                outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 64, num_beams =1)

            

            for i, (output, true) in enumerate(zip(outputs, labels)):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                
                try:
                    pred = pred.replace(" ,", ",").replace(". ", ", ").replace(".,", ",").replace("  "," ").replace(" ;",",").replace(" :", ",").replace("and", ",").strip()
                    pred = pred.replace("1.0", " 1.0").replace("1.5", " 1.5").replace("2.0", " 2.0").replace("2.5", " 2.5").replace("3.0", " 3.0").replace(
                        "3.5", " 3.5").replace("4.0", " 4.0").replace("4.5", " 4.5").replace("5.0", " 5.0")

                    if args.model_name == "bart":
                        pred_result = extract_traits(pred)
                    else:
                        preds = pred.split(",")
                        pred_result = dict()
                        for p in preds:
                            p = p.strip()
                            key, value = p.split(' ', 1)
                            pred_result[key] = float(value)
                    
                    true_result = "{" + re.sub(r'(\w+)\s([\d\.]+)', r'"\1": \2', true) + "}"
                    true_result = eval(true_result)


                    for trait in trait_list:
                        
                        pred_dic[trait].append(pred_result[trait])
                        true_dic[trait].append(true_result[trait])
                    
                except Exception as e:
                    
                    print(f"An error occurred: {e}")

                    # continue
                    break
        for trait in trait_list:
            try:
                qwk_result[trait] = quadratic_weighted_kappa(np.array(pred_dic[trait]), np.array(true_dic[trait]))
            except Exception as e:
                print(f"An error occurred: {e} for BART")
                traceback.print_exc() 
                qwk_result[trait] = 0.0
                                           
        log = "Test Result"
        log += f"\n| {qwk_result} |"
        print(log)

        

    return qwk_result, pred_dic, true_dic






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

# Tested 🕵🏻✅
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




def feedback_test(tokenizer, model, test_data, args):

    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
    trait_list = ["conventions", "grammar", "phraseology", "vocabulary", "syntax", "cohesion"]

   
    for trait in trait_list:
        pred_dic[trait] = list()
        true_dic[trait] = list()
        qwk_result[trait] = 0.0


    model.eval()
    batch_size = 128
    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i+batch_size]
            input_ids_all  = th.tensor(test['input_ids']).to(args.device)
            attention_mask =  th.tensor(test['attention_mask']).to(args.device)

            essay_input_ids = input_ids_all[:,:512]
            essay_attention_mask = th.ones(essay_input_ids.size(), dtype=th.long).to(model.device)

            
            if 'bart' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            elif 't5' in args.model_name:
                encoder_outputs = model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            elif 'pegasus' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            elif 'led' in args.model_name:
                encoder_outputs = model.led.encoder(input_ids=essay_input_ids,attention_mask=essay_attention_mask)
            
           
            criteria_ids = input_ids_all[:,512:]
            criteria_attention_mask = th.ones(criteria_ids.size(), dtype=th.long).to(model.device)
            if 'bart' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                
            elif 't5' in args.model_name:
                criteria_encoder_outputs = model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                
                
            elif 'pegasus' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                
            elif 'led' in args.model_name:
                criteria_encoder_outputs = model.led.encoder(input_ids=criteria_ids,attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.led.proj(th.concat([encoder_outputs[0],criteria_encoder_outputs[0]],dim=1).permute(0,2,1)).permute(0,2,1) 
                    
                
            labels = test['t5_output']

            
            decoder_start_token_id = model.config.decoder_start_token_id
            
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            
            if "flan-t5-base" in args.model_name:
                outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 256, num_beams =1)
            else:
                outputs = model.generate(input_ids=input_ids,encoder_outputs=encoder_outputs,max_new_tokens = 64, num_beams =1)

            

            for i, (output, true) in enumerate(zip(outputs, labels)):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                
                try:
                    pred = pred.replace(" ,", ",").replace(". ", ", ").replace(".,", ",").replace("  "," ").replace(" ;",",").replace(" :", ",").replace("and", ",").strip()
                    pred = pred.replace("1.0", " 1.0").replace("1.5", " 1.5").replace("2.0", " 2.0").replace("2.5", " 2.5").replace("3.0", " 3.0").replace(
                        "3.5", " 3.5").replace("4.0", " 4.0").replace("4.5", " 4.5").replace("5.0", " 5.0")

                    if args.model_name == "bart":
                        pred_result = extract_traits(pred)
                    else:
                        preds = pred.split(",")
                        pred_result = dict()
                        for p in preds:
                            p = p.strip()
                            key, value = p.split(' ', 1)
                            pred_result[key] = float(value)
                    
                    true_result = "{" + re.sub(r'(\w+)\s([\d\.]+)', r'"\1": \2', true) + "}"
                    true_result = eval(true_result)


                    for trait in trait_list:
                        
                        pred_dic[trait].append(pred_result[trait])
                        true_dic[trait].append(true_result[trait])
                    
                except Exception as e:
                    
                    print(f"An error occurred: {e}")

                    # continue
                    break
        for trait in trait_list:
            try:
                qwk_result[trait] = quadratic_weighted_kappa(np.array(pred_dic[trait]), np.array(true_dic[trait]))
            except Exception as e:
                print(f"An error occurred: {e} for BART")
                traceback.print_exc() 
                qwk_result[trait] = 0.0
                                           
        log = "Test Result"
        log += f"\n| {qwk_result} |"
        print(log)

        

    return qwk_result, pred_dic, true_dic

import re

def extract_traits(text):
    print("utils:extract_traits")
    matches = re.findall(r'(\w+)\s+([\d.]+)', text)
    
    
    traits_dict = {trait: float(value.rstrip('.')) for trait, value in matches[:7]} 
    return traits_dict








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

# Tested 🕵🏻✅
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
            print(f"💾 Saved top {rank+1} model to {model_path} with loss {loss:.4f}")

