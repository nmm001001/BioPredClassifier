from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

class TermEmbedder:
    def __init__(self, model_name='google-bert/bert-base-uncased'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        

    def _batch_term_embeds(self, terms, batch_size=32):
        term_embeddings = {}
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            encoded_dict = self.tokenizer(
                batch, 
                add_special_tokens=True,
                max_length=128, 
                padding='max_length',
                truncation=True, 
                return_attention_mask=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k,v in encoded_dict.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
            for term, embedding in zip(batch, batch_embeddings):
                term_embeddings[term] = embedding.squeeze()

        return term_embeddings

    def _embed_terms_list(self, term):
        encoded_dict = self.tokenizer(
            term, 
            add_special_tokens=True, 
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True, 
            return_tensors='pt'
        )
        inputs = encoded_dict.items()

        with torch.no_grad():
            output = self.model(inputs)

        embeds = output.last_hidden_state[:, 0, :].cpu().numpy()

        return embeds
        
    