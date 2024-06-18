import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import torch
import transformers
import json

json_file_path = r"./credentials.json"
with open(json_file_path, "r") as f:
    credentials = json.load(f)
newsapi = NewsApiClient(api_key=credentials["key"]) #crear un .json file para la key
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs = {
        "torch_dtype":torch.float16,
        "low_cpu_mem_usage":True,
    }
)
prompt_temp="""I'll give you some representative docs from a topic. You have to put a name to this topics, with a maximum length of 3 words. Your output must be this 3 words,without quotation marks. The representative docs are the following:"""
languages_dict= {
    "english":"en",
    "spanish":"es"
}

class News:
    def __init__(self, q ,language) -> None:
        self.q=q
        self.language=language

    def get_news(self):
        all_articles = newsapi.get_everything(q=self.q, language=languages_dict[self.language])
        return all_articles
    
    def clusterize_news(self):
        articles = self.get_news()
        articles['all']=articles['title'] + ' \n ' + articles['description']+ ' \n ' +articles['content']
        docs = articles['all'].values.tolist()
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_embeddings = embedding_model.encode(docs, show_progress_bar = True)
        vectorizer_model = CountVectorizer(stop_words = "english", max_df =.95, min_df = .01)
        # setting parameters for HDBSCAN (clustering) and UMAP (dimensionality reduction)
        hdbscan_model = HDBSCAN(min_cluster_size = 2, metric = 'euclidean',prediction_data = True)
        umap_model = UMAP(n_neighbors = 10, n_components = 10, metric ='cosine', low_memory = False)
        # Train BERTopic
        model = BERTopic(
            vectorizer_model = vectorizer_model,
            nr_topics = 'auto',
            top_n_words = 10,
            umap_model = umap_model,
            hdbscan_model = hdbscan_model,
            min_topic_size = 2, calculate_probabilities = True).fit(docs, corpus_embeddings)
        representative_docs=model.get_topic_info()['Representative_Docs'].values.tolist()
        return representative_docs
    
    def make_topics(self): 
        docs = self.clusterize_news()    
        topics = []
        for doc in docs:
            repr_doc='\n'.join(doc)
            prompt_entire=prompt_temp+repr_doc
            messages = [
                {'role':"system", "content":"You are a journalist"},
                {'role':"user", "content":prompt_entire},
            ]
            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt = True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens = 1024,
                eos_token_id = terminators,
                do_sample = True,
                temperature = 0.05,
                top_p = 0.9
            )
            topics.append(outputs[0]["generated_text"][len(prompt):])
            return topics #faltaria poner un diccionario o algo con topics y articles (o un dataframe)
                    