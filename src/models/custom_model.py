"""
One file containing the the training and evaluation for multiple transformer models.
"""
import sys
from transformers import (
    AutoModel, AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling,
    RobertaForMaskedLM, DistilBertForMaskedLM, RobertaModel
)
from transformers import get_scheduler
from transformers import TrainingArguments, Trainer
import numpy as np
import math

import os
os.environ['CURL_CA_BUNDLE'] = ''

sys.path.append("D:\\similarity-engine\\src\\")
sys.path.append("/home/rsaha/projects/similarity-engine/src/")

from utils.custom_dataset import CustomDataset

import torch
from torch.utils.data import DataLoader, RandomSampler

# Import optimizer and scheduler.
from torch.optim import AdamW


from tqdm import tqdm, trange

class CustomModel():
    def __init__(self, args, load_model_from_path=False, model_path=None, wandb_object=None):
        self.args = args
        self.device = args.device
        self.wandb_object = wandb_object

        if load_model_from_path:
            self.load_model(model_path)
            self.model.to(self.device)
        else:
            if self.args.model_name in ['bert-base-uncased', 'bert-large-uncased']:
                print("Loading pretrained model from HuggingFace Hub.")
                self.model = BertForMaskedLM.from_pretrained(self.args.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
            elif self.args.model_name in ['roberta-base', 'roberta-large', 'distilroberta-base']:
                self.model = RobertaForMaskedLM.from_pretrained(self.args.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
            elif self.args.model_name in ['distilbert-base-uncased']:
                self.model = DistilBertForMaskedLM.from_pretrained(self.args.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
            else:
                raise ValueError('Model not supported.')
    
    def preprocess_function(self, examples):
        return self.tokenizer([x for x in examples["merged"]], padding="max_length", truncation=True, max_length=512)
    
    def trainer_finetune(self, data, test_data, evaluate_only=False, evaluate_while_training=True):
        device = self.device
        model = self.model
        train_data = data.map(
            self.preprocess_function,
            batched=True,
            # num_proc=4,
            remove_columns=data.column_names,
        )
        test_data = test_data.map(
            self.preprocess_function,
            batched=True,
            # num_proc=4,
            remove_columns=test_data.column_names,
        )


        tokenizer = self.tokenizer
        # tokenizer.pad_token = tokenizer.eos_token
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")
        # train_dataset = CustomDataset(data, tokenizer)            
        train_dataset = train_data
        test_dataset = test_data
        # sampler = RandomSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=sampler)

        # Training loop.
        model = model.to(device)

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            evaluation_strategy="no" if not evaluate_while_training else "epoch",
            learning_rate=2e-5,
            num_train_epochs=self.args.epochs,
            weight_decay=0.01,
            push_to_hub=False,
            report_to="wandb",
            run_name=f"{self.args.model_name}_finetuned_{self.args.epochs}_epochs",
        )   
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        self.hf_trainer = trainer
        if not evaluate_only:
            self.hf_trainer.train()
        
        eval_results = self.hf_trainer.evaluate()
        print("Evaluate during training: ", evaluate_while_training)
        print("Evaluating at the end of training.")
        print("Evaluation results: ", eval_results)
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        self.wandb_object.log({"perplexity": math.exp(eval_results['eval_loss'])})
        
        return eval_results['eval_loss']


    def load_model(self, model_path):
        # Load the model.
        # model_path = self.args.load_model_path
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        
    
    def hf_trainer_evaluate(self, test_data):
        device = self.device
        model = self.model
        test_data = test_data.map(
            self.preprocess_function,
            batched=True,
            # num_proc=4,
            remove_columns=test_data.column_names,
        )

        tokenizer = self.tokenizer
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")
        test_dataset = test_data
        model = model.to(device)

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=self.args.epochs,
            weight_decay=0.01,
            push_to_hub=False,
        )   
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=test_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        self.hf_trainer = trainer
        
        eval_results = self.hf_trainer.evaluate()
        print("Evaluating only")
        print("Evaluation results: ", eval_results)
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        return eval_results['eval_loss']


    def train_model(self, data):

        device = self.device
        model = self.model
        
        # Check the model type.
        if self.args.model_type == 'bert':
            # Create a dataset for finetuning bert.

            train_dataset = CustomDataset(data, self.tokenizer)
            # val_dataset = CustomDataset(val_data, self.tokenizer)
            # test_dataset = CustomDataset(test_data, self.tokenizer)


            # Create a dataloader for the dataset.
            
            sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=sampler)
            # val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
            # test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

            num_training_steps = self.args.epochs * len(train_dataloader)

            # Create an optimizer.            
            optimizer = AdamW(model.parameters(), lr=self.args.lr)

            # Create a scheduler.
            scheduler = get_scheduler(
                'linear',
                optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )

        # Training loop.
        model = model.to(device)
        model.train()
        print("Training the model on device: ", device)
        train_iterator = trange(self.args.epochs, desc="Epoch", mininterval=0)
        
        for epoch in train_iterator:
            epoch_loss = 0.0
            train_iterator.set_description(f"Epoch {epoch + 1} of {self.args.epochs}")
            batch_iterator = tqdm(train_dataloader, desc=f"Running epoch {epoch + 1} of {self.args.epochs}", mininterval=0)
            for batch in batch_iterator:
                

                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs[0]
                current_loss = loss.item()
                batch_iterator.set_description(f"Running epoch {epoch + 1} of {self.args.epochs}, current loss: {current_loss}")
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


            # Print the epoch loss.
            print(f"Epoch {epoch + 1} loss: {epoch_loss / len(train_dataloader)}")
        
        print("Saving the model.")
        self.save_model(model)

            # Evaluate the model.
            # self.evaluate_model(val_dataloader, model)
    def save_model(self, model):
        model.save_pretrained(f"/home/rsaha/projects/similarity-engine/saved_models/{self.args.model_name}_finetuned_{self.args.epochs}_epochs/")
        

    def evaluate_model(self, val_dataset):
        device = self.device
        model = self.model
        model = model.to(device)
        model.eval()
        # print("Evaluating the model on device: ", device)
        val_dataset = val_dataset.map(self.preprocess_function, batched=True, remove_columns=val_dataset.column_names)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        val_iterator = tqdm(val_dataloader, desc="Evaluating the model", mininterval=0)
        val_losses = []
        for batch in val_iterator:
            batch = batch.to(device)
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            print("Loss: ", loss.item())
            val_losses.append(loss.item())
        print("Validation loss: ", np.mean(val_losses))
        return np.mean(val_losses)
            
            
    
    def get_embeddings(self, input_text, last_hidden_state=True):
        model = self.model
        model.to(self.device)
        model.eval()
        tokenized_embeddings = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        tokenized_embeddings = tokenized_embeddings.to(self.device)
        if last_hidden_state:
            # Tokenize embeddings.
            outputs = model(**tokenized_embeddings, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().detach().numpy()
            # print("CLS embedding shape: ", cls_embedding.shape)
            return cls_embedding
        else:
            pass