import sys
import os
sys.path.append(os.path.abspath(".."))
from source.RelationClassifier import RelationClassifier


# model_path = input('Enter model path: ')
classifier = RelationClassifier('../model/1000_pmid_model.pth')

print("Enter 'q' anytime to quit.\n")

while True:
    pmid = input('Enter PMID: ')
    if pmid.lower() == 'q':
        break

    term1 = input('Enter term 1: ')
    if term1.lower() == 'q':
        break

    term2 = input('Enter term 1: ')
    if term2.lower() == 'q':
        break

    score = classifier.get_score(pmid, term1, term2)

    print(f"Score: {score:.6f}")