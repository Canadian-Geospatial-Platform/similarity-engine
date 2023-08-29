from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # Tokenize all the examples here.
        # self.examples = []
        # for idx, row in self.data.iterrows():

        self.examples= [self.preprocess_data(d, self.tokenizer) for d in self.data['merged']]

    
    def __getitem__(self, idx):
        """
        Tokenizer the inputs here and to a batch encode.
        """
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)
    
    def preprocess_data(self, data, tokenizer):
        """
        Preprocess the data here.
        Create the input_ids, attention_mask, and labels.
        """
        batch = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='pt')
        return batch
        # input_ids = batch['input_ids'][0]
        # attention_mask = batch['attention_mask'][0]
        # token_type_ids = batch['token_type_ids'][0]
        # labels = batch['input_ids'][0].clone()

        # return {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     # 'token_type_ids': token_type_ids
        #     # 'labels': labels.type(torch.LongTensor)
        # }

        # batch = tokenizer(data, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        # input_ids = batch['input_ids'][0]
        # attention_mask = batch['attention_mask'][0]
        # token_type_ids = batch['token_type_ids'][0]
        # labels = batch['input_ids'][0].clone()

        # return {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'token_type_ids': token_type_ids,
        #     'labels': labels.type(torch.LongTensor)
        # }





        
