from HGCR_util.emb_lookup.abstr_emb_lookup import MedlineNumpyEmbeddings

class AbstractEmbeddingRetriever:
    def __init__(self):
        self.embeddings_fpath = Path(
            '/work/acslab/shared/medline_embeddings/PubMedNCL_abstr/abstr_id_to_emb_PubMedNCL'
        )
        self.pubmedncl_emb_obj = MedlineNumpyEmbeddings(
            emb_w_ids_fpath=embeddings_fpath,
            json_read_n_jobs=8,
        )

    def __getitem__(self, pmid):
        return self.pubmedncl_emb_obj[pmid]

        