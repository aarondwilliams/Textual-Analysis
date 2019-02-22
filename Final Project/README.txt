README - Exploration in Sentimental Analysis
-----------------------------------
Textual Analysis(ECE590.06)
Aaron Williams
12-16-2018

This is a Jupyter Notebook constructed to be used with a GPU in Python 3.

This Jupyter Notebook uses the following imports (-reason included):
numpy
	- Data handling
nltk
	- Dataset and other utility
matplotlib.pyplot
	- Graphs
string
	- Filtering Punctuation
tensorflow
	- For 2-layer NN and RNN
from nltk.corpus import wordnet
	- For lexicon expansion
from sklearn import svm
	- For SVM models
import gensim
	- For word embeddings
from nltk.data import find
	- To get pruned google model stored on nltk
	
All downloads are included within.
It should be possible to just run right through if you have the appropriate environment.

Note: I used some code that I developed for a HW assignment in Deep Learning.
	This in turn contains snippits from Deep Learning in-class code.

Warning: The RNNs takes some time (~1 hour each on my GPU) to train and test.