# Emotion Classifier — Reproducing Troiano, Padó & Klinger (ACL 2019)

A full-stack NLP app that classifies text into one of 7 emotions (anger,
disgust, fear, guilt, joy, sadness, shame), built to reproduce and extend the
modeling experiment from:

> Enrica Troiano, Sebastian Padó, and Roman Klinger. 2019. **Crowdsourcing and
> Validating Event-focused Emotion Corpora for German and English.** In
> *Proceedings of the 57th Annual Meeting of the Association for
> Computational Linguistics (ACL 2019)*, pages 4005–4011, Florence, Italy.
> [aclanthology.org/P19-1391](https://aclanthology.org/P19-1391/)

## What this project does

1. **Reproduces the paper's cross-corpus experiment (Table 4):** trains a
   MaxEnt-style classifier (L2-regularized logistic regression, boolean
   unigram features) on the original English ISEAR corpus, then evaluates it
   — with zero further training — on the paper's own enISEAR (English) and
   deISEAR (German, machine-translated) corpora.
2. **Extends beyond the paper:** trains a single bilingual model on all four
   corpora combined (ISEAR + enISEAR + deISEAR + de2enISEAR), so the live app
   can classify sentences typed in either English or German directly, with no
   translation step needed at inference time.
3. Wraps it in a small full-stack app: React frontend → Express API → Python
   inference script.

## Results vs. the paper

**English (trained on ISEAR, tested on enISEAR — unseen data):**

| emotion | ours (F1) | paper (F1) |
|---|---|---|
| anger | 0.19 | 0.27 |
| disgust | 0.45 | 0.45 |
| fear | 0.57 | 0.57 |
| guilt | 0.39 | 0.41 |
| joy | 0.69 | 0.67 |
| sadness | 0.56 | 0.58 |
| shame | 0.30 | 0.32 |
| **micro-F1** | **0.46** | **0.47** |

**German (trained on ISEAR, tested on deISEAR translated to English — unseen data):**

| emotion | ours (F1) | paper (F1) |
|---|---|---|
| anger | 0.25 | 0.29 |
| disgust | 0.48 | 0.49 |
| fear | 0.47 | 0.48 |
| guilt | 0.37 | 0.42 |
| joy | 0.60 | 0.68 |
| sadness | 0.46 | 0.53 |
| shame | 0.34 | 0.39 |
| **micro-F1** | **0.44** | **0.47** |

Both runs land within a few points of the paper's own numbers, and match its
key finding: joy is consistently the easiest emotion to classify, anger and
shame the hardest — in both languages.

## Architecture

App.jsx (React)  →  server.js (Express)  →  predict.py  →  emotion_model.pkl
`predict.py`, `server.js`, and `App.jsx` are fully model-agnostic — they just
load whatever's saved in `emotion_model.pkl` and forward text to it. Swapping
in a smarter or bilingual model never requires touching any of these three
files.

## Project structure

ISEAR.csv              original English ISEAR corpus (training data)
enISEAR.tsv             English cross-corpus test set (Troiano et al. 2019)
deISEAR.tsv             German cross-corpus test set, original language
de2enISEAR.tsv          deISEAR translated to English (by the paper's authors)
train.py                paper reproduction: train on ISEAR, evaluate on enISEAR
eval_german.py           paper reproduction: evaluate the same model on de2enISEAR
train_live.py            trains the combined bilingual model used by the live app
predict.py               loads emotion_model.pkl, classifies one sentence
server.js                Express API wrapping predict.py
frontend/                React app (Vite + Tailwind)

## Running it yourself

```bash
# 1. Install Python deps
pip install scikit-learn joblib pandas --break-system-packages

# 2. Reproduce the paper's English + German results
python train.py
python eval_german.py

# 3. Train the bilingual model used by the live app
python train_live.py

# 4. Start the backend
node server.js

# 5. Start the frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

## Data attribution

`enISEAR.tsv`, `deISEAR.tsv`, and `de2enISEAR.tsv` are from the official data
release accompanying Troiano, Padó & Klinger (2019), released under the
[Open Data Commons Attribution License (ODC-By) v1.0](https://www.opendatacommons.org/licenses/by/1.0/).
If you use this data, please cite the paper above.