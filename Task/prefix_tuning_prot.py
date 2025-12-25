import sys
sys.path.append('../')
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
from utils.log_helper import logger_init
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import logging
import random
from utils.set_seed import set_seed
import time


class TrainConfig:
    def __init__(self):
        
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_name_or_path =  os.path.join(self.project_dir, 'ProtGPT2') 
        self.tokenizer_name_or_path = os.path.join(self.project_dir, 'ProtGPT2') 
        self.train_path = os.path.join(self.project_dir, 'data', 'Allalpha', 'train_alpha.csv')
        self.test_path = os.path.join(self.project_dir, 'data', 'Allalpha', 'test_alpha.csv')
        # self.train_path = os.path.join(self.project_dir, 'data', 'AMP', 'train_amp.csv')
        # self.test_path = os.path.join(self.project_dir, 'data', 'AMP', 'test_amp.csv')
        self.num_virtual_tokens = 100   # Allalpha:100 / AMP:20 
        self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=self.num_virtual_tokens)
        self.task_type = 'prefix'   # prefix / finetune
        self.dataset_name = "Allalpha"  # Allalpha / AMP
        self.text_column = "Sequence"

        self.max_length = 512  # 512  # Allalpha:512 / AMP:200 
        self.lr = 5e-3   # Allalpha: ft:5e-6  pt:0.005 / AMP: ft:1e-6  pt:0.001 
        self.num_epochs = 100
        self.batch_size = 8   # Allalpha:8 / AMP:32 
        self.random_seed = 42

        self.model_save_dir = os.path.join(self.project_dir, 'cache', f'{self.dataset_name}', f'{self.task_type}')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs', f'{self.dataset_name}', f'{self.task_type}')

        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)

        if self.task_type == 'prefix':
            self.writer = SummaryWriter(f"runs/{self.dataset_name}/{self.task_type}_vt/ploysche_{self.peft_config.peft_type}_{self.peft_config.task_type}_E{self.num_epochs}_LR{self.lr}_BS{self.batch_size}_ML{self.max_length}_VT{self.num_virtual_tokens}")  
            logger_init(log_file_name=f"ploysche_{self.dataset_name}_{self.task_type}_E{self.num_epochs}_LR{self.lr}_BS{self.batch_size}_ML{self.max_length}_VT{self.num_virtual_tokens}", log_level=logging.INFO, log_dir=self.logs_save_dir)
    
        else:
            self.writer = SummaryWriter(f"runs/{self.dataset_name}/{self.task_type}/ploysche_finetune_{self.peft_config.task_type}_E{self.num_epochs}_LR{self.lr}_BS{self.batch_size}_ML{self.max_length}")  
            logger_init(log_file_name=f"ploysche_{self.dataset_name}_{self.task_type}_E{self.num_epochs}_LR{self.lr}_BS{self.batch_size}_ML{self.max_length}", log_level=logging.INFO, log_dir=self.logs_save_dir)
        logging.info(" ### Print the current configuration to a log file ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    data_files = {"train": config.train_path, "test": config.test_path}
    dataset = load_dataset('csv', data_files=data_files)
    logging.info('check the info about dataset')
    logging.info(dataset)
    logging.info(dataset["train"][0])
    logging.info(dataset["test"][0])


    # data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path) 

    if tokenizer.pad_token_id is None:   # eos
        tokenizer.pad_token_id = tokenizer.eos_token_id


    def preprocess_function(examples):
        batch_size = len(examples[config.text_column])  
        print(batch_size)
        inputs = [x for x in examples[config.text_column]]   
        model_inputs = tokenizer(inputs)   # tokenize
        labels = tokenizer(inputs)
        # each sample
        for i in range(batch_size):
            sample_input_ids = [tokenizer.eos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
            label_input_ids = [tokenizer.eos_token_id] + labels["input_ids"][i] + [tokenizer.eos_token_id]
            labels["input_ids"][i] = label_input_ids
            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        #padding
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id] * (      # add pad
                config.max_length - len(sample_input_ids)
            )
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] +  [0] * (config.max_length - len(sample_input_ids)) 
            labels["input_ids"][i] = label_input_ids + [-100] * (config.max_length - len(sample_input_ids))
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:config.max_length])   # [:max_length]
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:config.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:config.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # dataset
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    logging.info(processed_datasets)

    # dataset
    # split_dataset = processed_datasets["train"].train_test_split(train_size=0.95, seed=42)
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    logging.info(train_dataset)
    logging.info(eval_dataset)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logging.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 3):
        logging.info(f"Sample {index} of the test set: {eval_dataset[index]}.")

    # dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

    # creating model
    if config.task_type == 'prefix':
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        model = get_peft_model(model, config.peft_config)
        model.print_trainable_parameters()
    elif config.task_type == 'finetune':
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

    logging.info(model)
    if config.task_type == 'prefix':
        logging.info(model.peft_config)

    # model
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs), lr_end=1e-7, power=3
    )

    # training and evaluation
    model = model.to(config.device)
    time_start = time.time()
    for epoch in range(config.num_epochs):  # num_epochs
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, ncols=50)):
            global_iter_num = epoch * len(train_dataloader) + step + 1
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            config.writer.add_scalar('train_b/Loss', loss.item(), global_iter_num)
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader, ncols=50)):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        # logging.info(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        config.writer.add_scalar('train_e/Loss', train_epoch_loss, epoch)
        config.writer.add_scalar('train_e/ppl', train_ppl, epoch)
        config.writer.add_scalar('eval_e/Loss', eval_epoch_loss, epoch)
        config.writer.add_scalar('eval_e/ppl', eval_ppl, epoch)
    time_end = time.time()
    time_sum = time_end - time_start
    config.writer.add_scalar('training time', time_sum, 0)
    logging.info(f'The sum time of training {time_sum}')

    # saving model
    if config.task_type == 'prefix':
        peft_model_id = config.model_save_dir + f"/prefix_{config.peft_config.peft_type}_{config.peft_config.task_type}_E{config.num_epochs}_LR{config.lr}_BS{config.batch_size}_ML{config.max_length}_VT{config.num_virtual_tokens}"
        model.save_pretrained(peft_model_id)
    elif config.task_type == 'finetune':
        finetune_model_id = config.model_save_dir + f"/finetune_E{config.num_epochs}_LR{config.lr}_BS{config.batch_size}_ML{config.max_length}"
        model.save_pretrained(finetune_model_id)


if __name__ == '__main__':
    train_config = TrainConfig()
    set_seed(train_config.random_seed)
    train(train_config)
