from HGCR_util import text_util
from HGCR_util.lazy_json_kv_loader import LazyJsonlAbstractLoader


class AbstractRetriever:
    def __init__(self):
        self.sent_jsonl_dir = Path(
            '/work/acslab/shared/Agatha_shared/'
            'pmid_to_sent_id_w_text_kv_jsonl_lazy_chunks'
        )

        self.abstr_db = LazyJsonlAbstractLoader(
            list(sent_jsonl_dir.glob('*jsonl'))
        )

    def __getitem__(self, pmid):
        return text_util.get_abstr_text(pmid, self.abstr_db)

    