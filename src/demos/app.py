import gradio as gr
from transformers import AutoTokenizer, BertForMaskedLM
import torch


def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'bert' in model_name:
        model = BertForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError('Model not supported.')
    return tokenizer, model


def predict(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    token_logits = model(input_ids)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    top_5_tokens_str = tokenizer.convert_ids_to_tokens(top_5_tokens)
    return top_5_tokens_str




with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML('<h1>Choose Embedding type and Clustering Algorithm</h1>')
    with gr.Row():
        with gr.Column():
            embedding_type = gr.Radio(['Word2Vec - Average', 'BERT - Average', 'BERT - CLS'], label='Embedding type', info='Choose the embedding type.', value='BERT - CLS')
        with gr.Column():
            clustering_algorithm = gr.Radio(['KMeans', 'Agglomerative Hierarchical - single linkage', 'DBSCAN', 'HDBSCAN'], label='Clustering Algorithm', info='Choose the clustering algorithm.', value='KMeans')
    gr.HTML('<h1>Choose Hyperparameters</h1>')
    with gr.Row():
        with gr.Column():
            seed = gr.Number(value=42, label='Seed (For KMeans)', info='Choose the seed for the clustering algorithm.')
            n_clusters = gr.Number(label='Number of clusters (For KMeans and single linkage hierarchical)', info='Choose the number of clusters.', value=3)
        with gr.Column():
            with gr.Row():
                eps = gr.Number(value=0.5, label='Epsilon (For DBSCAN)', info='Choose the epsilon for DBSCAN.')
                min_samples = gr.Number(value=5, label='Min Samples (For DBSCAN)', info='Choose the min_samples for DBSCAN.')
            min_cluster_size = gr.Number(value=5, label='Min Cluster Size (For HDBSCAN)', info='Choose the min_cluster_size for HDBSCAN.')
    with gr.Row():
        metric = gr.Dropdown(['euclidean', 'manhattan', 'cosine'], label='Metric', info='Choose the metric for the clustering algorithm.', value='euclidean')
    gr.HTML('<h1>Results</h1>')
    with gr.Row():
        with gr.Column():
            silhouette = gr.Textbox(label='Silhouette Score')
        with gr.Column():
            adjusted_rand = gr.Textbox(label='Adjusted Rand Score')
        with gr.Column():
            purity = gr.Textbox(label='Purity')
    # gr.HTML('<h1>Visualization in 2D</h1>')
    # with gr.Row():
    #     true_plots = gr.ScatterPlot(x='x', y='y', value=df)
    #     cluster_plots = gr.ScatterPlot(x='x', y='y', value=pd.DataFrame)

    btn = gr.Button('Run clustering')
    # btn.click(fn=run_clustering, inputs=[embedding_type, clustering_algorithm, seed, n_clusters, eps, min_samples, min_cluster_size, metric], outputs=[silhouette, adjusted_rand, purity])#, true_plots, cluster_plots])




demo.launch()