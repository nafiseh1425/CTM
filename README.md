# CTM
Introduction
This project showcases a topic modeling approach that combines Bag of Words (BoW) with Sentence-BERT (SBERT) embeddings. Topic modeling is a technique used to discover the main themes or topics within large collections of documents. By identifying these topics, we can gain insights into the content without needing to read each document individually. This makes topic modeling a valuable tool for researchers and analysts working with extensive textual data.

Key Features
Topic Modeling: Automatically identifies recurring themes in a set of documents.
Combined Approach: Uses both BoW and SBERT to improve the coherence and relevance of topics.
Applications: Useful in natural language processing, information retrieval, and content recommendation systems.
Data Preparation
The dataset used for training and validation is sourced from Wikipedia. To begin, download the necessary files:

'!wget https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_unprep.txt
!wget https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_prep.txt


