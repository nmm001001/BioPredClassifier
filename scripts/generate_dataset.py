import numpy
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath('..'))
import pandas as pd
import random
from transformers import AutoModel, AutoTokenizer
from utils.SemrepOutputLoader import SemrepOutputLoader
from utils.TermEmbedder import TermEmbedder
# from utils.AbstractEmbeddingRetriever import AbstractEmbeddingRetriever
# from utils.AbstractRetriever import AbstractRetriever
from HGCR_util.lazy_json_kv_loader import LazyJsonlAbstractLoader
from sklearn.model_selection import train_test_split
from HGCR_util import text_util
from HGCR_util.emb_lookup.abstr_emb_lookup import MedlineNumpyEmbeddings
from tqdm import tqdm
random.seed(42)

## parameters
pkl_filename = "/work/acslab/shared/rel_extraction/semrep_data/data_2021_onw/agatha_medline_sentences/pubmed25n1270_sentences.pkl"
num_samples = 50000
negatives_per_positive = 5
absract_term_threshold = 3
model_name = "google-bert/bert-base-uncased"

embeddings_fpath = Path(
    '/work/acslab/shared/medline_embeddings/PubMedNCL_abstr/abstr_id_to_emb_PubMedNCL'
)

pubmedncl_emb_obj = MedlineNumpyEmbeddings(
    emb_w_ids_fpath=embeddings_fpath,
    json_read_n_jobs=8,
)

sent_jsonl_dir = Path(
    '/work/acslab/shared/Agatha_shared/'
    'pmid_to_sent_id_w_text_kv_jsonl_lazy_chunks'
)

abstr_db = LazyJsonlAbstractLoader(
    list(sent_jsonl_dir.glob('*jsonl'))
)

sr_output_dict = SemrepOutputLoader('/work/acslab/shared/rel_extraction/semrep_data/data_2021_onw/agatha_medline_sentences')
sr_output_dict._load_single_pkl_file('pubmed25n1270_sentences.pkl')

sampled_pmids = random.sample(list(sr_output_dict.data_dict.keys()), num_samples)
sampled_pmids = [i for i in sampled_pmids if len(sr_output_dict[i]['relations']) > absract_term_threshold]

sampled_abstracts = [text_util.get_abstr_text(pmid, abstr_db) for pmid in tqdm(sampled_pmids, desc="Retrieving Abstracts...")]
sampled_embeds = []
for pmid in tqdm(sampled_pmids, desc="Retrieving Abstract embeddings..."):
    try:
        sampled_embeds.append(pubmedncl_emb_obj[pmid])
    except Exception as e:
        sampled_embeds.append(None)

sr_terms = {pmid: sr_output_dict[pmid].get('terms') for pmid in tqdm(sampled_pmids, desc="Retrieving SemRep Terms")}
sr_pairs = {pmid: sr_output_dict[pmid].get('relations', []) for pmid in tqdm(sampled_pmids, desc="Retrieving SemRep Pairs...")}

all_terms = list(set(term for terms in sr_terms.values() for term in terms))

term_embedder = TermEmbedder()
term_embeddings = term_embedder._batch_term_embeds(all_terms)

pmid_term_embeddings = {pmid: [term_embeddings[term] for term in sr_terms[pmid]] for pmid in sampled_pmids}

pmid_pair_embeddings = {pmid: [(term_embeddings[p1], term_embeddings[p2]) for p1, p2 in pairs if p1 in term_embeddings and p2 in term_embeddings] for pmid, pairs in sr_pairs.items()}

df = pd.DataFrame({
    "pmid": sampled_pmids,
    "abstract": sampled_abstracts,
    "abstract_embeddings": sampled_embeds,
    "terms": [list(sr_terms[pmid]) for pmid in sampled_pmids],
    "term_embeddings": [[term_embeddings[term] for term in sr_terms[pmid]] for pmid in sampled_pmids],
    "relations": [list(sr_pairs[pmid]) for pmid in sampled_pmids],
    "relations_embeddings": [pmid_pair_embeddings[pmid] for pmid in sampled_pmids]
})

df.to_pickle(f"../data/{num_samples}_pmid_dataset.pkl")

def generate_negatives(row, negatives_per_positive=negatives_per_positive):
    """ Generate negative pairs for each abstract """
    terms = row['terms']
    existing_pairs = set(row['relations'])
    negatives = set()

    num_positives = len(existing_pairs)
    num_negatives = negatives_per_positive * num_positives
    
    while len(negatives) < num_negatives:
        t1, t2 = random.sample(terms, 2)
        if (t1, t2) not in existing_pairs and (t1, t2) not in negatives:
            negatives.add((t1, t2))

    return list(negatives)

def generate_negative_pair_emebddings(row, term_embeddings):
    return [(term_embeddings[p1], term_embeddings[p2]) for p1, p2 in row['negative_relations'] if p1 in term_embeddings and p2 in term_embeddings]


df['negative_relations'] = df.apply(generate_negatives, axis=1)
df['negative_relations_embeddings'] = df.apply(
    lambda row: generate_negative_pair_embeddings(row, term_embeddings)
)

df.to_pickle(f"../data/{num_samples}_pmid_dataset.pkl")









