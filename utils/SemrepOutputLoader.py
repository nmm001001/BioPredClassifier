import pickle
from tqdm import tqdm
import os
import mmap
from collections import defaultdict
from joblib import Parallel, delayed
import itertools

class SemrepOutputLoader:
    def __init__(self, file_dir, max_threads=8):
        self.file_dir = file_dir
        self.max_threads = max_threads
        self.data_dict = defaultdict(list)
        
    def _load_pkl_files(self, file):
        try:
            with open(file, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    data = pickle.loads(mmapped_file.read())
                    return self._parse_pkl_files(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return []

    def _load_pkl_files_parallel(self):
        files = [os.path.join(self.file_dir, file) for file in os.listdir(self.file_dir) if file.endswith('.pkl')]

        results = Parallel(n_jobs=self.max_threads)(delayed(self._load_pkl_files)(file) for file in tqdm(files, desc="Loading SemRep outputs...."))

        for parsed_data in results:
            for d in parsed_data:
                pmid = d['pmid']
                self.data_dict[pmid]['terms'].update(d['terms'])
                self.data_dict[pmid]['relations'].update(d['relations'])

    def _load_single_pkl_file(self, file):
        file_path = os.path.join(self.file_dir, file)
        parsed_data = self._load_pkl_files(file_path)
        for sent_dict in parsed_data:
            pmid = sent_dict['pmid']
            self.data_dict[pmid].append({
                'terms': sent_dict['terms'],
                'relations': sent_dict['relations'],
                'negatives': sent_dict['negatives']
            })

    def _parse_pkl_files(self, data):
        parsed_data = []

        for d in data:
            pmid = d.get('pmid')
            relations = d.get('relations', [])
            terms = d.get('terms', [])

            if len(relations) == 0:
                continue

            term_texts = [t.get('extracted_text') for t in terms if t.get('extracted_text')]
            if len(term_texts) < 2:
                continue

            pos_pairs = set()
            for r in relations:
                subj, obj = r.get('subj_text'), r.get('obj_text')
                if subj and obj:
                    pos_pairs.add(tuple(sorted((subj, obj))))

            all_pairs = set(tuple(sorted(p)) for p in itertools.combinations(term_texts, 2))
            neg_pairs = all_pairs - pos_pairs

            parsed_data.append({
                'pmid': pmid, 
                'terms': set(term_texts),
                'relations': pos_pairs,
                'negatives': neg_pairs,
            })
        return parsed_data
            

    def __getitem__(self, pmid):
        return self.data_dict[pmid]

        


    

        
                
        

    