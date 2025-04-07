import os
import argparse
import torch as th
from utils import *
from models.customized_modeling_t5 import CustomizedT5ForConditionalGeneration
from transformers import T5Tokenizer
import pickle
import warnings

warnings.filterwarnings("ignore")

def main(args):
    """
    📝 General Notes:

    Dynamic arguments 
    args.result_path: asap/t5-base  -- saves results?
    args.save_model_path: ckpts_asap/t5-base -- saves checkpoints?
    args.device: cpu -- what it does idk 

    Used Emojis 🙃:
     ✅ 🆘 🕵🏻 📌 👀 🔄 🛠️ 😊 😮‍💨 🚶 📍 📶 💾 💡 🔧 🤔 📟 📃 🚀 🏋️‍♂️ 📤 🚨 🧠 🗂️ 🟡 🔵 🧮 🏗️

    
    """
    print("🦾 THE BIG BOSS")
    set_seed(args)

    # 🕵🏻✅ Malak: Creates ckpts_asap folder with model name folder inside if not created 
    if not os.path.isdir(f"ckpts_{args.result_path}"):
        os.makedirs(f"ckpts_{args.result_path}")
        print(f"🗂️ New folder created: ckpts_{args.result_path}")

    # 🕵🏻✅ Malak: Creates new argument called save_model_path ckpts_asap/t5-base 
    args.save_model_path = f"ckpts_{args.result_path}"


    # 🕵🏻🆘 To be tested : if in test mode it modifies args.load_checkpoint_path to point to a checkpoint directory where model checkpoints are saved
    if args.test:
        args.load_checkpoint_path = f"ckpts_{args.result_path}"
        print(f"In Test Mode: { args.load_checkpoint_path}")

    
    # 🕵🏻✅ Malak: It always assigns gpu i tried -1/1/0
    # If a GPU is available and the user hasn't explicitly set gpu=-1, it assigns a specific CUDA device. Otherwise, it defaults to CPU.
    if th.cuda.is_available() and args.gpu != -1:
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    # Load T5 tokenizer
    print("📟 T5 tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    if args.data == "asap":
         print("ASAP Add additional tokens")
         add_tokens = ["@", "{", "}",'<essay>',"<rationale>","[overall]", "[content]", "[organization]", "[word choice]", "[sentence fluency]", "[conventions]","[prompt adherence]", "[language]", "[narrativity]", "[style]","[voice]", 
                 "overall", "content", "organization", "word choice", "sentence fluency", "conventions", "prompt adherence", "language", "narrativity", "style", "voice"]
    else:
        print("Feedback Add additional tokens")
        add_tokens = ["@", "{", "}",'<essay>',"<rationale>","[cohesion]", "[syntax]", "[vocabulary]", "[phraseology]", "[grammar]", "[conventions]", "1.0", "1.5", 
                      "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "conventions", "grammar", "vocabulary", "phraseology", "syntax", "cohesion"]
    
    tokenizer.add_tokens(add_tokens)

    # These dictionaries will store results, predictions, and ground truths for each fold in cross-validation.
    best_fold_result_dict = dict()
    best_fold_pred_dict = dict()
    best_fold_true_dict = dict()
    sub_best_fold_result_dict = dict()
    sub_best_fold_pred_dict = dict()
    sub_best_fold_true_dict = dict()

# The function performs 5-fold cross-validation, training and evaluating a separate model for each fold.
# 💡 From what I understand LoRA helps in datasets where data is not too much or not to little so try decreasing folds if results are not good

    for fold in range(1):
        print(f"🚀 Begining of Fold Number:{fold}")
        model = CustomizedT5ForConditionalGeneration.from_pretrained(args.model_name)
        model.use_rationale = True

        model.resize_token_embeddings(len(tokenizer))
        
        # Creates a directory for saving models specific to this fold. // ckpts_asap/ creates files with each fold 
        save_model_fold_path = os.path.join(args.save_model_path, str(fold))
        if not os.path.isdir(save_model_fold_path):
            os.makedirs(save_model_fold_path)
        args.save_model_fold_path = save_model_fold_path

        # Reads train, dev, and test data for the current fold.
        # Applies preprocess_data function to convert the raw text into tokenized input.
        if args.data == "asap":
            TRAIN_DATA_PATH = f"./data/essay/fold_{fold}/train.csv"
            DEV_DATA_PATH = f"./data/essay/fold_{fold}/dev.csv"
            TEST_DATA_PATH = f"./data/essay/fold_{fold}/test.csv"
        else:
            TRAIN_DATA_PATH = f"./data/feedback/fold_{fold}/train.csv"
            DEV_DATA_PATH = f"./data/feedback/fold_{fold}/dev.csv"
            TEST_DATA_PATH = f"./data/feedback/fold_{fold}/test.csv"

        # This function likely reads the CSV file and returns the data in a format (such as a Pandas DataFrame or a dataset object) that can be processed further.
        # read_data is a function in utils
        train_data = read_data(TRAIN_DATA_PATH)
        dev_data = read_data(DEV_DATA_PATH)
        test_data = read_data(TEST_DATA_PATH)
        """
        🕵🏻✅ Malak: For testing reading of data : it reads data correctly + length of enteries are correct
        # Print number of entries
        print(f"\nTotal number of entries: {len(train_data)}")


        # Print first entry with each column on a separate line
        first_entry = train_data[0]
        print("\nFirst entry:")
        for col, val in first_entry.items():
          print(f"{col}: {val}")
        """
        train_dataset = train_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        dev_dataset = dev_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)
        test_dataset = test_data.map(lambda x: preprocess_data(x, tokenizer,args), batched=True)


        """
        🕵🏻✅ Malak: For testing processing of data : it reads processes data correctly 

        #Checking the encoded entry 
        print(tokenizer.decode(train_dataset["input_ids"][0]))
        print(tokenizer.decode(train_dataset["labels"][0]))

        
        # Number of examples processed
        print(len(train_dataset["input_ids"])) 
        print(len(train_dataset["labels"])) 

        📌 Summary of Function (preprocess_data)
 
        Tokenize Essay and Rationale: The essay and rationale (either GPT or Llama) are tokenized separately and then concatenated.

        Tokenize Labels: The output labels (like scores) are tokenized based on the model type.

        Combine Data: Both the essay input and rationale are merged into a single input sequence, and labels are prepared for the model.
        x represents a batch of examples

        
        The batched= True argument means that preprocess_data will be applied to the data in batches, not one sample at a time, which is more efficient.

        """
        if not args.test:
            print(f"🏋️‍♂️ Training Fold : {fold}")
            model = train(model, tokenizer, train_dataset, dev_dataset, args)

# Loads the best model checkpoint (if available) and evaluates it on the test set.
            print(f"{args.save_model_fold_path}")
            for filename in os.listdir(args.save_model_fold_path):
                if filename.startswith("checkpoint-1"):
                    
                    best_model_path = os.path.join(args.save_model_fold_path, filename)
                    best_checkpoint = th.load(best_model_path)
                    model.load_state_dict(best_checkpoint)
                    best_model = model.to(args.device)
        
                    if args.data == "asap":
                        best_result, best_pred_dic, best_true_dic = asap_test(tokenizer, best_model, test_dataset, args)
                    else:
                        best_result, best_pred_dic, best_true_dic = feedback_test(tokenizer, best_model, test_dataset, args)
                    
# Moves the model to CPU, deletes it, clears CUDA cache, and runs garbage collection to free memory.
                    
                    best_model = best_model.cpu()
        
                    del best_model
                    th.cuda.empty_cache()
                    gc.collect()  
        
                elif filename.startswith("checkpoint-2"):
                    sub_best_model_path = os.path.join(args.save_model_fold_path, filename)
                    sub_best_checkpoint = th.load(sub_best_model_path)
                    model.load_state_dict(sub_best_checkpoint)
                    sub_best_model = model.to(args.device)
                    if args.data == "asap":
                        sub_best_result, sub_best_pred_dic, sub_best_true_dic = asap_test(tokenizer, sub_best_model, test_dataset, args)
                    else:
                        sub_best_result, sub_best_pred_dic, sub_best_true_dic = feedback_test(tokenizer, sub_best_model, test_dataset, args)
                    sub_best_model = sub_best_model.cpu()
                    
                    del sub_best_model
                    th.cuda.empty_cache()
                    gc.collect()  

# If in test mode, it skips training and directly loads pre-trained checkpoints for evaluation.
        elif args.test:
            print(f"Model Test Fold : {fold}")
            for filename in os.listdir(args.save_model_fold_path):
                if filename.startswith("checkpoint-1"):
                    best_model_path = os.path.join(args.save_model_fold_path, filename)
                    best_checkpoint = th.load(best_model_path)
                    model.load_state_dict(best_checkpoint)
                    best_model = model.to(args.device)

                    if args.data == "asap":
                        best_result, best_pred_dic, best_true_dic = asap_test(tokenizer, best_model, test_dataset, args)
                    else:
                        best_result, best_pred_dic, best_true_dic = feedback_test(tokenizer, best_model, test_dataset, args)
                    best_model = best_model.cpu()
        
                    del best_model
                    th.cuda.empty_cache()
                    gc.collect()  
        
                elif filename.startswith("checkpoint-2"):
                    sub_best_model_path = os.path.join(args.save_model_fold_path, filename)
                    sub_best_checkpoint = th.load(sub_best_model_path)
                    model.load_state_dict(sub_best_checkpoint)
                    sub_best_model = model.to(args.device)
                    if args.data == "asap":
                        sub_best_result, sub_best_pred_dic, sub_best_true_dic = asap_test(tokenizer, sub_best_model, test_dataset, args)
                    else:
                        sub_best_result, sub_best_pred_dic, sub_best_true_dic = feedback_test(tokenizer, sub_best_model, test_dataset, args)
                    
                    sub_best_model = sub_best_model.cpu()
                    
                    del sub_best_model
                    th.cuda.empty_cache()
                    gc.collect() 

# Saves the best and sub-best results for this fold.
        best_fold_result_dict[fold] = best_result
        best_fold_pred_dict[fold] = best_pred_dic
        best_fold_true_dict[fold] = best_true_dic
        
        
        sub_best_fold_result_dict[fold] = sub_best_result
        sub_best_fold_pred_dict[fold] = sub_best_pred_dic
        sub_best_fold_true_dict[fold] = sub_best_true_dic
        
        with open(f"./{args.result_path}/best_result_dict.pkl", "wb") as f:
            pickle.dump(best_fold_result_dict, f)
        with open(f"./{args.result_path}/best_pred_dict.pkl", "wb") as f:
            pickle.dump(best_fold_pred_dict, f)
        with open(f"./{args.result_path}/best_true_dict.pkl", "wb") as f:
            pickle.dump(best_fold_true_dict, f)
        with open(f"./{args.result_path}/sub_best_result_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_result_dict, f)
        with open(f"./{args.result_path}/sub_best_pred_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_pred_dict, f)
        with open(f"./{args.result_path}/sub_best_true_dict.pkl", "wb") as f:
            pickle.dump(sub_best_fold_true_dict, f)
        
# Returns dictionaries containing predictions and ground truths for analysis.     
    return best_fold_result_dict, best_fold_pred_dict, best_fold_true_dict, \
        sub_best_fold_result_dict, sub_best_fold_pred_dict, sub_best_fold_true_dict
        
        


if __name__ == "__main__":

# The script starts by defining command-line arguments using argparse.ArgumentParser. This allows users to configure parameters when running the script.
        
        parser = argparse.ArgumentParser('Essay Scoring')
        parser.add_argument('--gpu', '-g', type=int, default=0, help='which gpu to use, specify -1 to use CPU')
        parser.add_argument('--train_batch_size', '-trb', type=int, default=4, help='batch_size')
        parser.add_argument('--test_batch_size', '-teb', type=int, default=128, help='test_batch_size')
        parser.add_argument('--seed', '-s', type=int, default=40, help='random seed')
        parser.add_argument('--patience', '-p', type=int, default=10, help='number of patience for early stopping')
        parser.add_argument("--train_epochs", type=int, default=15)
        parser.add_argument("--save_checkpoint_path", type=str, default=None) # !! not used 
        parser.add_argument("--test", type=bool, default=False)
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--data", type=str, default=f"asap")
        parser.add_argument("--llm", type=str, default="gpt")
        parser.add_argument('--model_name', '-m', type=str, default='t5-base', help='name of the t5 model')
        parser.add_argument('--learning_rate', '-l', type=float, default=5e-05, help='learning rate')
        
        args = parser.parse_args()    
        
        # Malak: Creates asap/feedback folder and model name folder inside it 
        args.result_path = os.path.join(args.data, args.model_name.replace('/', '_'))
        os.makedirs(args.result_path, exist_ok=True)
        

        # Run main function (trains & evaluates the model)
        best_fold_result_dict = main(args)

        # # Save final results using Pickle
        # with open(f"./{args.result_path}/final_best_result_dict.pkl", "wb") as f:
        #     pickle.dump(best_fold_result_dict, f)

    
