## TASK: Fine-tune LLaMA-3.2-3B using LoRA for Q&A Task

# INPUT: 
- model_name = "meta-llama/Llama-3.2-3B-Instruct"         # CHOSED INSTRUCT MODEL TO FINE TUNE USING QA DATASET
- dataset_file = "covid.json"                             # QA DATASET PROVIDED
- target_gpu_memory = 16GB                                # ASSUMED THE AVILABLE GPU RESOURCES

# STEP 1: ENVIRONMENT SETUP
    Install required packages:
        - transformers, datasets, peft, accelerate, trl
        - torch, bitsandbytes for model quantization
        - huggingface_hub for model upload
        - evaluate, rouge_score for metrics

# STEP 2: DATA PREPROCESSING
    FUNCTION preprocess_data(dataset_file):
        Load JSON data from dataset_file
        FOR each data sample:
            Extract question, context, answer
            Format using chat template:
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>
                Question: {question}\nContext: {context}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>"
            Validate and clean data
        Split data into train/validation sets (80/20)
        Apply tokenization with max_length=512
        RETURN processed_dataset

# STEP 3: MODEL CONFIGURATION
    FUNCTION setup_model_and_tokenizer():
        Load tokenizer from model_name
        Configure BitsAndBytesConfig for 4-bit quantization:  
            - load_in_4bit = True
            - bnb_4bit_compute_dtype = torch.float16
            - bnb_4bit_use_double_quant = True
            - bnb_4bit_quant_type = "nf4"
        Load base model with quantization config
        Enable gradient checkpointing for memory efficiency
        Prepare model for k-bit training
        RETURN model, tokenizer

# STEP 4: DATA COLLATOR SETUP         
    FUNCTION setup_data_collator(tokenizer):
        Configure DataCollatorForLanguageModeling:
            - tokenizer = tokenizer
            - mlm = False (causal language modeling)
            - pad_to_multiple_of = 16 (GPU efficiency for 16GB)
            - return_tensors = "pt"
        RETURN data_collator

# STEP 5: LORA CONFIGURATION
    FUNCTION setup_lora_config():
        Configure LoraConfig:   
            - r = 16 
            - lora_alpha = 32 
            - target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]
            - lora_dropout = 0.05
            - bias = "none"
            - task_type = "CAUSAL_LM"
        Apply LoRA to model using get_peft_model()
        Print trainable parameters count
        RETURN peft_model

# STEP 6: TRAINING CONFIGURATION
    FUNCTION setup_training_args():
        Configure TrainingArguments:
            - output_dir = "./llama-covid-qa-lora"
            - num_train_epochs = 3
            - per_device_train_batch_size = 4  # Optimized for 16GB
            - per_device_eval_batch_size = 4
            - gradient_accumulation_steps = 4  
            - warmup_steps = 100
            - learning_rate = 2e-4
            - weight_decay = 0.01
            - fp16 = True
            - logging_steps = 10
            - save_strategy = "epoch"
            - evaluation_strategy = "epoch"
            - load_best_model_at_end = True
            - metric_for_best_model = "eval_loss"
            - greater_is_better = False
        RETURN training_args

# STEP 7: TRAINING LOOP
    FUNCTION train_model(model, tokenizer, dataset, training_args):
        data_collator = setup_data_collator(tokenizer)
        Initialize SFTTrainer with:
            - model = peft_model
            - train_dataset = processed_train_data
            - eval_dataset = processed_eval_data
            - tokenizer = tokenizer
            - data_collator = data_collator 
            - args = training_args
            - max_seq_length = 512
            - packing = False
        
        Execute trainer.train()
        Save final model
        RETURN trained_model

# STEP 8: EVALUATION
    FUNCTION evaluate_model(trainer):
        Execute trainer.evaluate()
        Calculate additional metrics:
            - Perplexity = exp(eval_loss)
            - Token accuracy if available
        Log and return evaluation results
        RETURN evaluation_metrics

# STEP 9: MODEL MERGING
    FUNCTION merge_lora_with_base(peft_model):
        Merge LoRA adapters with base model:
            - merged_model = peft_model.merge_and_unload()
        Save merged model locally
        RETURN merged_model

# STEP 10: MODEL UPLOAD
    FUNCTION push_to_huggingface(model, tokenizer, repo_name):
        Login to Hugging Face Hub
        Create repository if not exists
        Push merged model and tokenizer to hub:
            - model.push_to_hub(repo_name)
            - tokenizer.push_to_hub(repo_name)
        RETURN upload_status

# STEP 11: MODEL TESTING
    FUNCTION test_merged_model(merged_model, tokenizer, test_samples):
        FOR each test sample:
            Generate response using merged model
            Compare with ground truth
            Calculate similarity metrics
        RETURN test_results

# MAIN EXECUTION:
    1. model, tokenizer = setup_model_and_tokenizer()
    2. preprocessed_data = preprocess_data("covid.json", tokenizer)
    3. peft_model = setup_lora_config(model)
    4. training_args = setup_training_args()
    5. trainer = train_model(peft_model, tokenizer, preprocessed_data, training_args)
    6. eval_results = evaluate_model(trainer)
    7. merged_model = merge_lora_with_base(trainer.model)
    8. test_results = test_merged_model(merged_model, tokenizer, test_samples)
    9. upload_status = push_to_huggingface(merged_model, tokenizer, "hf_username/llama-covid-qa")

# OUTPUT: 
    - Fine-tuned LoRA model saved locally
    - Merged model uploaded to Hugging Face Hub  
    - Evaluation and test results logged
