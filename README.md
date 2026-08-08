# Week 6 - Advanced NLP: NER, Topic Modeling & Transfer Learning

AI & ML Internship - Task 6

## Project Overview

Three advanced NLP techniques applied to 2,225 real BBC news articles across five categories (business, entertainment, politics, sport, tech). Part 1 uses Named Entity Recognition (spaCy) to extract people, organizations and locations and analyse which entity types dominate each category. Part 2 uses Topic Modeling (LDA) to discover hidden themes and compare them to the real news sections. Part 3 uses Transfer Learning to classify articles with a pre-trained transformer (feature extraction) and benchmarks it against a classic TF-IDF baseline. Together these mirror the NLP pipelines behind media analytics, financial intelligence, and news aggregation.

## Dataset

BBC News - 2,225 articles across five categories. The notebook downloads it automatically from a stable public mirror (suraj-deshmukh/BBC-Dataset-News-Classification, latin-1 encoded), so no manual download is needed. Original source: Kaggle (jacopoferretti/bbc-articles-dataset).

## Part 1 - Named Entity Recognition (spaCy)

The spaCy en_core_web_sm pipeline tags entities such as PERSON, ORG, GPE, DATE and MONEY. On this run PERSON (10,391) and ORG (9,265) dominate, followed by DATE (7,179), GPE (6,488) and NORP (2,420). Politics and business are richest in ORG and GPE; sport and entertainment lean on PERSON. A heatmap breaks entity frequency down by category, and five example articles are shown with their extracted entities.

## Part 2 - Topic Modeling (LDA)

Latent Dirichlet Allocation is run with five topics to test whether an unsupervised model rediscovers the five supervised categories. It does: the topic-vs-category cross-tabulation is an almost perfect diagonal - sport, entertainment, politics, tech and business each map onto their own discovered topic, with 92 to 98 percent of each category concentrated on a single topic. Model quality is reported with perplexity (1875.2) and log-likelihood.

## Part 3 - Transfer Learning

The approach is feature extraction: the pre-trained sentence-transformer all-MiniLM-L6-v2 embeds each article into a 384-dimensional vector, and a Logistic Regression head is trained on those embeddings. A classic TF-IDF + Logistic Regression model is the baseline. On a 20 percent stratified test split (445 articles) the Transformer + LR scores accuracy 0.973 and macro-F1 0.972, while TF-IDF + LR scores accuracy 0.978 and macro-F1 0.977. Both are excellent; the best model is saved as bbc_classifier.joblib.

## Repository Structure

notebooks/week6_advanced_nlp.ipynb is the executed notebook with all outputs visible. data/ holds dataset notes. README.md and requirements.txt complete the repo.

## How to Run

Open notebooks/week6_advanced_nlp.ipynb in Google Colab and choose Runtime then Run all. The dataset downloads automatically and all six charts plus the saved model are regenerated.

## Tools

Python, pandas, NumPy, scikit-learn (LDA, TF-IDF, LogisticRegression), spaCy (NER), sentence-transformers (MiniLM), Matplotlib, Seaborn, joblib.
