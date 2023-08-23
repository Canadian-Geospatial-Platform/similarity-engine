import gradio as gr
from transformers import AutoTokenizer, BertForMaskedLM
import torch
import gensim
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize




def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if 'pretrained' in model_name:
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    elif 'finetuned' in model_name:
        path = "/home/rsaha/projects/similarity-engine/saved_models/trainer_bert_fine_tune_full_data/checkpoint-4000/"
        model = BertForMaskedLM.from_pretrained(path)
    elif 'word2vec' in model_name:
        path = "/home/rsaha/projects/similarity-engine/saved_models/word2vec/word2vec.model"
        model = gensim.models.Word2Vec.load(path)
    return tokenizer, model




def normalize_inputs(embeddings):
    return normalize(embeddings)[0]

def load_df_with_similar_records(path):
    return pd.read_csv(path)

def get_all_fine_tuned_bert_embeddings(path):
    return np.load(path)['arr_0']


def get_similar_records_for_record(search, num_similar_records=5):
    tokenizer, model = get_model("pretrained")
    similar_df_path = "/Users/simpleparadox/PycharmProjects/similarity-engine/data/train_fine_tune_full_data_sim_matrix_no_embeds.csv"
    similar_records_df = load_df_with_similar_records(similar_df_path)
    fine_tuned_embeds_path = "/Users/simpleparadox/PycharmProjects/similarity-engine/data/all_fine_tuned_bert_embeds.npz"
    all_fine_tuned_bert_embeds = get_all_fine_tuned_bert_embeddings(fine_tuned_embeds_path)
    print("Search: ", search)
    num_similar_records = int(num_similar_records)
    if num_similar_records > 10:
        num_similar_records = 10
    cls_embedding = get_embeddings(model, tokenizer, search)

    # Now calculate the similarity between the embedding of the search query and the embeddings of all the records.
    # First normalize the embeddings.
    # cls_embedding = normalize_inputs(cls_embedding)
    # all_fine_tuned_bert_embeds = normalize_inputs(all_fine_tuned_bert_embeds)
    print("CLS embedding shape: ", cls_embedding.shape)
    similarity_scores = cosine_similarity(cls_embedding.reshape(1,-1), all_fine_tuned_bert_embeds)

    # Now get the top 20 indices for the records with the highest similarity scores.
    top_indices = np.argsort(similarity_scores[0])[::-1][:num_similar_records]

    top_records = similar_records_df.iloc[top_indices]

    return top_records, {'1':0, '2':0, '3':0, '4':0, '5':0}




def get_embeddings(model, tokenizer, text, last_hidden_state=True):
    model.eval()
    tokenized_embeddings = tokenizer(text, return_tensors="pt", truncation=True)
    if last_hidden_state:
        outputs = model(**tokenized_embeddings, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().detach().numpy()
        # print("CLS embedding shape: ", cls_embedding.shape)
        return cls_embedding




with gr.Blocks() as demo:
    print("Model loaded")
    # print(model)
    # model_name = gr.Radio(["BERT-pretrained", "BERT-finetuned", "Word2Vec-scratch"], label="Model", value="BERT-pretrained")
    search = gr.Textbox(label="Search Query")
    num_similar_records = gr.Textbox(label="Number of similar records", value='5')
    with gr.Row():
        label_preds = gr.DataFrame(label="Predicted Records", interactive=True)
        similar_records_preds = gr.Label(label="Similar records")

    greet_btn = gr.Button(label="Search")
    greet_btn.click(fn=get_similar_records_for_record, inputs=[search, num_similar_records], outputs=[label_preds, similar_records_preds])

demo.launch()