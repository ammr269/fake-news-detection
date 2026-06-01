# Fake News Detection : Hybrid Multimodal System

Final year project (PFE) - Master in Computer Science  
Mohamed Soilahoudine

---

## Overview

This project implements a hybrid system for fake news detection that combines three complementary modules: textual analysis with BERT, source credibility scoring, and knowledge graph embeddings. The system is evaluated on the GossipCop subset of FakeNewsNet.

---

## Dataset

The original FakeNewsNet dataset provides article titles and URLs for two sources: PolitiFact (political fact-checking) and GossipCop (celebrity news). Since the dataset does not include article bodies, a web scraping pipeline was built to retrieve full texts.

The scraping yielded highly asymmetric results. Over 10,000 GossipCop articles were successfully retrieved, while PolitiFact returned fewer than 100 accessible articles due to paywalls and removed URLs. This made a full-text evaluation on PolitiFact infeasible.

The final working dataset is GossipCop, balanced to 5,152 articles (2,576 fake / 2,576 real) after deduplication and quality filtering. A separate title-only PolitiFact dataset of 862 articles was kept for auxiliary experiments on Module 2.

The dataset file `fakenewsnet_gossipcop_balanced.csv` is included in the repository.

---

## Architecture

The system is structured as three sequential modules that can be used independently or fused:

**Module 1 - Textual analysis**  
Baseline: TF-IDF + SVM (two versions: title only, title + text)  
Deep model: BERT (bert-base-uncased), fine-tuned on the classification task

**Module 2 - Source credibility scoring**  
Each article is scored along five dimensions: MBFC credibility score for the source domain, uppercase ratio in the title, exclamation/question mark density, sensational word count, and external link count. The resulting vector is concatenated to the BERT [CLS] representation before classification.

**Module 3 - Knowledge graph embeddings**  
Named entities are extracted with spaCy (PERSON, ORG, GPE, LOC, EVENT). A co-occurrence graph is built from the corpus itself, with two relation types: `co_occurs_with` (two entities appearing in the same article) and `has_type` (entity to semantic category). TransE is trained on this graph via PyKEEN (64 dimensions, 50 epochs). The resulting entity embeddings are averaged per article and concatenated to the BERT and source vectors in the final trimodal model.

Note on Wikidata: the Wikidata API returned HTTP 403 from the Colab training environment. The intra-corpus graph is the adopted solution. It produces structurally valid embeddings specific to the GossipCop domain.

---

## Results

All experiments use the same 80/20 stratified split (random_state=42).

| Model                                | Accuracy | F1     | AUC    |
| ------------------------------------ | -------- | ------ | ------ |
| TF-IDF + SVM (title only)            | 0.7876   | 0.7875 | 0.8599 |
| TF-IDF + SVM (title + text)          | 0.8293   | 0.8293 | 0.9066 |
| BERT (title only)                    | 0.7808   | 0.7804 | 0.8625 |
| BERT (title + text)                  | 0.7973   | 0.7972 | 0.8955 |
| BERT + Source — M1+M2 (title only)   | 0.8235   | 0.8234 | 0.8942 |
| BERT + Source — M1+M2 (title + text) | 0.8118   | 0.8118 | 0.8929 |
| M1+M2+M3 TransE (title + text)       | 0.8147   | 0.8147 | 0.8943 |
| M1+M2+M3 TransE (title only)         | 0.7924   | 0.7924 | 0.8685 |

Auxiliary result on PolitiFact (title only, 862 articles):  
BERT + Source M1+M2 — Accuracy 0.9017 | F1 0.9016 | AUC 0.9602

The TF-IDF + SVM baseline with full text outperforms BERT in all title-only configurations, which reflects the lexically distinctive nature of GossipCop articles. BERT + Source (title only) achieves the best F1 on the main dataset. The addition of the KG module provides a marginal gain over BERT alone, consistent with the corpus-internal nature of the graph.

---

## Repository Structure

```
.
├── data/
│   └── fakenewsnet_gossipcop_balanced.csv
├── notebooks/
│   ├── 00_download_and_cleaning.ipynb
│   ├── 01_module1_tfidf.ipynb
│   ├── 02_module1_bert.ipynb
│   ├── 03_module2_with_text.ipynb
│   ├── 04_module2_title_only.ipynb
│   └── 05_module3_graph_embeddings.ipynb
└── README.md
```

Cache files (TransE embeddings, KG vectors) are stored in Google Drive and loaded automatically when running the notebooks. Paths are set at the top of each notebook.

---

## Dependencies

```
torch
transformers
pykeen
spacy (en_core_web_sm)
scikit-learn
pandas
numpy
matplotlib
requests
```

All notebooks run on Google Colab with a GPU runtime (T4 or A100). Average training time per module is 15 to 30 minutes depending on the configuration.

---

Université Mohamed premier, Oujda

---

## License

Academic use only.
