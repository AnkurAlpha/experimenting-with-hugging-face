# ---------------------------------------------------------------------------------
# I am trying to cache the weights as it is taking so much time
# to load again and again in each run. This is just a experiment.
# Do not do like this on real production, pipeline objects are fragile.
# ---------------------------------------------------------------------------------
from transformers import pipeline
import pickle
from pathlib import Path

# cache path
cache_path = Path("cachefile") / "distilbert_classifier.pkl"

# make the parent directory if not exists
if not cache_path.parent.exists():
    cache_path.parent.mkdir()

if cache_path.exists():
    classifier = pickle.load(open(cache_path, "rb"))
else:
    classifier = pipeline("sentiment-analysis",
                          model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    pickle.dump(classifier, open(cache_path, "wb"))

result = classifier(
    "Hey there! I have been soo much exited for you to open this repo!")
print(result)

"""
 ╭─ankur@ankur in repo: experimenting_with_huggingface/pipeline_funcitons on  main [!?] via  v3.14.3 (venv) took 0s
 ╰─λ python 01_basics.py
Loading weights: 100%|██████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 18176.07it/s]
[{'label': 'NEGATIVE', 'score': 0.8496776819229126}]

 ╭─ankur@ankur in repo: experimenting_with_huggingface/pipeline_funcitons on  main [!?] via  v3.14.3 (venv) took 5s
 ╰─λ bat 02_reuse_weights1.py

 ╭─ankur@ankur in repo: experimenting_with_huggingface/pipeline_funcitons on  main [!?] via  v3.14.3 (venv) took 1s
 ╰─λ python 02_reuse_weights1.py
[{'label': 'NEGATIVE', 'score': 0.8496776819229126}]

 ╭─ankur@ankur in repo: experimenting_with_huggingface/pipeline_funcitons on  main [!?] via  v3.14.3 (venv) took 3s
 ╰─λ

"""
