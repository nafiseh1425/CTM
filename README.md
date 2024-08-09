# CTM
Introduction
This project showcases a topic modeling approach that combines Bag of Words (BoW) with Sentence-BERT (SBERT) embeddings. Topic modeling is a technique used to discover the main themes or topics within large collections of documents. By identifying these topics, we can gain insights into the content without needing to read each document individually. This makes topic modeling a valuable tool for researchers and analysts working with extensive textual data.

Key Features
Topic Modeling: Automatically identifies recurring themes in a set of documents.
Combined Approach: Uses both BoW and SBERT to improve the coherence and relevance of topics.
Applications: Useful in natural language processing, information retrieval, and content recommendation systems.
Data Preparation
The dataset used for training and validation is sourced from Wikipedia. To begin, download the necessary files:
```
!wget https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_unprep.txt
!wget https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_prep.txt

Installation
To install the required libraries, run the following commands:

!pip install contextualized-topic-models
!pip install torch torchvision

Loading Libraries
The following libraries are used in this project:

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file, TopicModelDataPreparation
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamodel
import os
import numpy as np
import pickle

Data Handling
We start by loading the preprocessed and unprocessed text files. These files are used to create training datasets:

with open("dbpedia_sample_abstract_20k_prep.txt", 'r') as fr_prep:
  text_training_preprocessed = [line.strip() for line in fr_prep.readlines()]

with open("dbpedia_sample_abstract_20k_unprep.txt", 'r') as fr_unprep:
  text_training_not_preprocessed = [line.strip() for line in fr_unprep.readlines()]


Note: Ensure that the lengths of the preprocessed and unprocessed documents are the same.

Training Dataset Creation
To create the training dataset, we use the TopicModelDataPreparation class, which handles both BoW creation and SBERT embeddings:

tp = TopicModelDataPreparation("bert-base-nli-mean-tokens")
training_dataset = tp.fit(training_contextual_document, training_bow_documents)


tp = TopicModelDataPreparation("bert-base-nli-mean-tokens")
training_dataset = tp.fit(training_contextual_document, training_bow_documents)

Combined Topic Model
The CombinedTM model is used to perform topic modeling by combining BoW with SBERT embeddings. This approach increases the coherence of the predicted topics:

ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, num_epochs=100, n_components=50)
ctm.fit(training_dataset)

Saving and Loading the Model
You can save the trained model for future use:

ctm.save(models_dir="./")

To load a saved model:

ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, num_epochs=100, n_components=50)
ctm.load("/kaggle/working/contextualized_topic_model_nc_50_tpm_0.0_tpv_0.98_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99/", epoch=99)


Topic Analysis
Once the model is trained, you can analyze the topics:

ctm.get_topic_lists(5)

Predicting Topics for Unseen Documents
You can use the model to predict topics for new, unseen documents:

testing_dataset = tp.fit(testing_contextual_documents, testing_bow_documents)
predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=10)


To get the topic predictions for a specific document:

topic_index = np.argmax(predictions[15])
ctm.get_topic_lists(5)[topic_index]



References
Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). Cross-lingual Contextualized Topic Models with Zero-shot Learning. European Chapter of the Association for Computational Linguistics (EACL). Link to Paper
GitHub Repository




