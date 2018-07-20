
# coding: utf-8

# # DH 2018 Machine Reading: Advanced Topics in Word Vectors

# ## **Welcome to Machine Reading!**
# 
# This is a hands-on workshop focusing on various word vectorization methods and applications for digital humanities.
# The workshop will be split into 4 sections with 10 minute breaks in-between. The sections get incrementally more advanced, building on concepts and methods from the previous sections.
# 
# To follow along, you can run the script portions piecemeal, in order, as we progress through the workshop material.
# 
# Instructors:
# - Eun Seo Jo, <a href="mailto:eunseo@stanford.edu">*eunseo@stanford.edu*</a>, Stanford University, Stanford Literary Lab
# - Javier de la Rosa, <a href="mailto:versae@stanford.edu">*versae@stanford.edu*</a>, Stanford University
# - Scott Bailey, <a href="mailto:scottbailey@stanford.edu">*scottbailey@stanford.edu*</a>, Stanford University
# 
# 
# ## 1. Understanding Word Vectors with Visualization (50 mins)
# 
# This unit will give a brief introduction of word vectors and word embeddings. Concepts needed to understand the internal mechanics of how they work will also be explained, with the help of plots and visualizations that are commonly used when working with them.
# 
# 
# - 0:00 - 0:20 Sparse and dense vectors (SVD, PMI, etc.)
# - 0:20 - 0:35 What to do with vectors (cosine similarity, etc.)
# - 0:35 - 0:50 Visualizations (Clustering, PCA, t-SNE) 
# 
# 1. What are the limitations of these word vectors?
# 2. What are the different use cases between count-based vectors and word2vec? (size of corpus)
# 3. What are limitations?
# 4. Why do we use Word2Vec instead?
# 
# ## 2. Word Vectors via Word2Vec (50 mins)
# 
# This unit will focus on Word2Vec as an example of neural net-based approaches of vector encodings, starting with a conceptual overview of the algorithm itself and ending with an activity to train participants’ own vectors.
# 
# 
# - 0:00 - 0:15 Conceptual explanation of Word2Vec
# - 0:15 - 0:30 Word2Vec Visualization and Vectorial Features and Math
# - 0:30 - 0:50 Word2Vec Construction [using Gensim] and Visualization (from part 1)
# 
# 
# ## 3. Pre-trained Models and Extended Vector Algorithms (50 mins)
# 
# This unit will explore the various flavors of word embeddings specifically tailored to sentences, word meaning, paragraph, or entire documents. We will give an overview of pre-trained embeddings, including where they can be found and how to use them.
# 
# - 0:00 - 0:15 [Out-of-vocabulary words and pre-trained embeddings](part3.ipynb#1.-Out-of-vocabulary-words-and-pre-trained-embeddings)
# - 0:15 - 0:25 [Activity] Bias in pre-trained historical word embeddings
# - 0:25 - 0:40 [Extending Vector Algorithms: Text Classification](part3.ipynb#2.-Extending-Vector-Algorithms:-Text-Classification)
# - 0:40 - 0:50 [Activity] Authorship attribution
# 
# ## 4. Role of Bias in Word Embeddings (50 mins)
# 
# In this unit, we will explore an application and caveat of using word embeddings -- cultural bias. Presenting methods and results from recent articles, we will show how word embeddings can carry the historical biases of the training corpora and lead an activity that shows these human-biases on vectors. We'll also address how such bias can be mitigated.
# 
# - 0:00 - 0:10 Algorithmic bias vs human bias 
# - 0:10 - 0:40 Identifying bias in corpora (occupations, gender, ...) 
# - 0:40 - 0:50 Towards unbiased embeddings; Examine “debiased” embeddings
# - 0:50 - 0:60 Concluding remarks and debate
# 

# # 0. Setting Up 

# Before we get started, let's go ahead and set up our notebook.
# 
# We will start by importing a few Python libraries that we will use throughout the workshop.
# 
# ## What are these libraries?
# 
# 1. NumPy: This is a package for scientific computing in python. For us, NumPy is useful for vector operations. 
# 2. NLTK: Easy to use python package for text processing (lemmatization, tokenization, POS-tagging, etc.)
# 3. matplotlib: Plotting package for visualization
# 4. scikit-learn: Easy to use python package for machine learning algorithms and preprocessing tools
# 5. gensim: Builtin word2vec and other NLP algorithms
# 
# We will be working with a few sample texts using NLTK's corpus package.

# In[1]:


get_ipython().run_cell_magic('capture', '--no-stderr', 'import sys\n!pip install Cython\n!pip install -r requirements.txt\n!python -m nltk.downloader all\nprint("All done!", file=sys.stderr)\n\n#! = escape out of Jup')


# If all went well, we should be able now to import the next packages into our workspace

# In[2]:


import numpy as np
import nltk
import sklearn # science kit = machine learning tool
import matplotlib.pyplot as plt # plotting tool
import gensim #machine learning 


# 
# 
# ---
# 
# 

# # 1. Understanding Word Vectors with Visualization
# 
# 

# ## What is a word vector?
# 
# A word vector or embedding is a **numerical representation** of a word within a corpus based on co-occurence with other words. Linguists have found that much of the meaning of a word can be derived from looking at its surrounding context. In this unit, we will explore a few major approaches to representing words in a numerical format.

# ## What is a vector?
# 
# Before anything related to words or text let's make sure we're on the same page about vectors! A vector is just a list/array of real numbers. A vector has a size/length which indicates how many numbers are in it. In Python you can make a vector using square brackets '[]'.

# In[3]:


# 
vector_one = [1, 2, 3]
vector_two = [1, 2, 34.53222222]
vector_three = [-2494, 3, 48.2934, -0.49484]


# Here is a list of 5 real numbers (represented as floating point numbers). This vector has 5 dimensions or features. Unlike formal vectors, Python lists can contain different types of elements and do not support vector operations broadly. NumPy provides a numerical engine with proper vector/array implementations.

# In[4]:


# Here you can generate a vector of random floats with the random function from numpy
# You'll see that every time you run this command you get a series of different numbers - try it!
# In this instance we're making a vector of length (or size) 5

vector_of_floats = np.random.randn(5)
vector_of_floats


# Here is a list of 20 integers between 0 and 3 (exclusive; not including 3). Later we will go into more vector math but you can see that a vector is a multi-dimensional numerical representation.

# In[5]:


# You can call a vector of random integers too
# There are three inputs here: the start range of your integer, 
# the end range(exclusive), and the size of the vector
# In our example, the range is [0, 3)

vector_of_ints = np.random.randint(0, 3, size=20)
vector_of_ints


# In[21]:


# Activity: Try making vectors of your own here!
my_vector = np.random.randint(5, 55, size=300)
my_vector


# Word vectors (and vectors in general) can be largely classified into **sparse** (=veel nullen) and **dense** vectors.
# 
# A sparse vector is count-based vector where each element in the vector represent the integer counts of words, usually co-occurence or frequecy. Because a lot of words don't appear all the time, many elements of sparse vectors are 0, to represent 0 observations. (altijd veel nullen omdat most numbers have a very low change of co-occurrence with most other words)
# 
# There are a few examples of sparse vectors we will examine here. 

# ## Document-term matrix
# 
# One of the simplest and most common ways of constructing a matrix is recording its occurence throughout a set of documents. This creates a document-term matrix where one dimension indicates the frequency of a word in documents and the other indicates the vocabulary (all words that occur at least once in your entire corpus).

# Among the many packages that help you construct your own matrix with your corpus, `scikit-learn` is one of the most heavily used within the Python scientific stack. Let's import `scikit-learn`'s `CountVectorizer()` 

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer


# In[7]:


# Imagine you have a document that is just a sentence like this...

documents = [
    "This is a piece of text. This is also some random text. Also text.",
]


# Let's now transform this document so that each word is given a unique identifying number.

# In[8]:


example_vectorizer = CountVectorizer() #initialize your count vectorizer
example_vectorizer.fit(documents) #documents much be a vector of strings(individual documents)
print("Vocabulary size:", len(example_vectorizer.vocabulary_))
example_vectorizer.vocabulary_  #We can get the unique vocabulary set and its corresponding index here


# Also, we add the corresponding frequency number, which gives the total number of times each word appears in each document.

# In[9]:


counts = example_vectorizer.transform(documents)
print(counts)
print("   ↑  ↑         ↑\n  doc word_id count")


# Now, let's iterate through all the words that appear in our original document and print all the counts that we generated above.

# In[28]:


doc = 0  # first document
for word, word_id in example_vectorizer.vocabulary_.items():
    print(word, ":", counts[doc, word_id])


# A **document-term matrix** is just a big table (formally, a mathematical matrix) that describes the frequency of words or terms that occur in a collection of documents. In a document-term matrix, **rows correspond to documents** in the collection and **columns correspond to terms**. In 
# 
# In our case, since we only have one document, our document-term matrix only has one row (doc `0`) and looks like this.
# 
# |             | also | is | of | piece | random | some | text | this |
# | ----------- |:----:|:--:|:--:|:-----:|:------:|:----:|:----:|:----:|
# | Document #0 |  2   | 2  | 1  | 1     |  1     |  1   |  3   |   2  |
# 
# It can easily be extracted by using the `transform()` method of our `CountVectorizer()`.

# In[29]:


counts = example_vectorizer.transform(documents)
counts.toarray()


# Each element of the matrix represents vocabulary from above, with the placement corresponding to the unique identifier assigned by scikit-learn, eg. 7th placement (6th, starting from 0) is `text`.

# Let's now add a new document that looks almost identical but introduces a new word, just to see how this change reflects on the document-term matrix.

# In[31]:


documents = [
    "This is a piece of text. This is also some random text. Also text.",
    "This is a piece of text. This is also some random text. Also new text.",
]
example_vectorizer.fit(documents)
print("Vocabulary size:", len(example_vectorizer.vocabulary_))
example_vectorizer.vocabulary_


# In[33]:


counts = example_vectorizer.transform(documents)
print(counts)
print("   ↑  ↑         ↑\n  doc word_id count")


# In[34]:


counts = example_vectorizer.transform(documents)
counts.toarray()


# Now with two documents our matrix looks like this. 
# 
# |             | also | is | new | of | piece | random | some | text | this |
# | ----------- |:----:|:--:|:---:|:--:|:-----:|:------:|:----:|:----:|:----:|
# | Document #0 |  2   | 2  |  0  | 1  | 1     |  1     |  1   |  3   |   2  |
# | Document #1 |  2   | 2  |  1  | 1  | 1     |  1     |  1   |  3   |   2  |

# Apart from the fact that the vocabulary size is now bigger, just by looking at the document-term matrix we can easily spot that one of our documents has one word more than the other. Certainly, we can spot the difference at column 3 (2 in zero-index Python sequences), which corresponds to the new word introduced, `new`, in our vocabulary. You can see there is also an additional column for the additional document (document 2). You can induce what the matrix would look like with lots more documents and a bigger vocabulary!

# By now, you might have noticed that 1-letter words are being ignored. That's due to the way `CountVectorizer()` splits sentences into words. `CountVectorizer()` has options to customize this behaviour and it allows to specify your own regular expression to extract words, disregard stopwords, count ngrams instead of words, cap the max number of words to count, normalize spelling, or count terms within a frequency range. It is worth exploring the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

# Here, we have written up a temporary new regular expression that takes into account 1-letter words, so our `CountVectorizer()` can count 'a' as a vocabulary term. As such you can modify the regex to fix you needs.

# In[36]:


new_regex = r"(?u)\b\w+\b"  # this regex now considers single character tokens
CountVectorizer(token_pattern=new_regex).fit(documents).vocabulary_


# In[40]:


#Activity: Make your own corpus of documents and see what kind of doc-term matrix you can generate!

my_corpus = ['this is our trial document','hello good bye random word blue green purple','i wonder if other people feel the same', 'new','new','new']
my_vectorizer = CountVectorizer()
my_vectorizer.fit(my_corpus) 
my_vectorizer.transform(my_corpus).toarray()


# OK, since we have vectorized lots of text humans have generated we will now turn to some canons. 
# 
# **Let's now play with three texts/documents in our corpus, taken from literature. ** 
# 
# We will use Moby Dick, Emma, and Parents as our example texts in our corpus.
# Each text is treated as a document.

# In[41]:


from nltk.corpus import gutenberg  


# In[46]:


mobydick = gutenberg.raw('melville-moby_dick.txt')
emma = gutenberg.raw('austen-emma.txt')
parents = gutenberg.raw('edgeworth-parents.txt')


# In[48]:


corpus = [mobydick, emma, parents]


# In[49]:


# We do the same thing as above 
lit_vectorizer = CountVectorizer(token_pattern=new_regex)
lit_vectorizer.fit(corpus)
print("Vocabulary size:", len(lit_vectorizer.vocabulary_))
lit_vectorizer.vocabulary_


# To get the ID of a given vocab term:

# In[50]:


print("The ID of the word 'piece' is ", str(lit_vectorizer.vocabulary_.get('piece')))


# In[51]:


X = lit_vectorizer.fit_transform(corpus)


# This is what the doc-term matrix looks like for our three document corpus.

# In[52]:


X = X.toarray()
X #Remember each row corresponds to each document (novel) and each column is each word from our combined vocabulary


# The dimensions of the matrix are given by the shape property.

# In[53]:


X.shape # How many novels are there?   # How big is our vocabulary?


# In[54]:


print("The doc-term matrix has {} documents and {} dimensions.".format(str(X.shape[0]), str(X.shape[1])))


# In[55]:


# You can look up all the words in the vocab from the three novels
lit_vectorizer.get_feature_names() 


# Let's get vocab IDs for 'happy', 'sad', 'angry'

# In[56]:


print(lit_vectorizer.vocabulary_.get('happy'))
print(lit_vectorizer.vocabulary_.get('sad'))
print(lit_vectorizer.vocabulary_.get('angry'))


# In[58]:


X[:,8860], X[:,16233], X[:,1059] 
#This is one way of making word vectors. 
#What kind of information do you think these vectors represent?


# In[60]:


#In another instance, you could treat each sentence as one document

from nltk import sent_tokenize
sentences = []
for novel in ['melville-moby_dick.txt', 'austen-emma.txt', 'edgeworth-parents.txt']:
    sentences += sent_tokenize(gutenberg.raw(novel))
len(sentences)


# In[61]:


# sentences as documents now
lit_vectorizer = CountVectorizer(token_pattern=new_regex)
X = lit_vectorizer.fit_transform(sentences).toarray()
X.shape


# In[62]:


X #Here you will notice most of these elements are zeros! Why?


# In[63]:


#happy, sad, angry
X[:,8860], X[:,16233], X[:,1059] 


# In[64]:


np.set_printoptions(threshold=np.inf)
print(X[:,8860]) # let's see all the zeros!


# In[68]:


print('There are '+str(len(X[:,8860]))+' total elements since there were this many sentneces total')


# In[69]:


print("... and of these " + str(len(np.where(X[:,8860]>0)[0])) + " have non-zero entries.")
#What does that mean about this word?


# In[70]:


print("In total this word appears " + str(np.sum(X[:,8860])) + " times.")
#What does that say about this word?


# Doc-term matrices are used in information retrieval for finding documents that are most relevant to a query. If you look at each row (rather than column) you get a numerical representation of a document by the words that appear in it. 

# In[71]:


#Just putting this back normal print options
np.set_printoptions(threshold=10)


# ## Word-word Matrix
# 
# In the previous section we looked at representing words by their relations to a corpus of documents. What about their relation to one another? The most intuitive way of doing this is to build a word-word matrix where now both dimensions are the vocab and each element at [**k**][**l**] represents the co-occurence of the vocab **k** with vocab **l** in a window of **w**. The window of **w** indicates the number of words before and after given word **k** where we count occurrneces of **l**. **w** is usually around 4. 

# In[72]:


from nltk.tokenize import word_tokenize
import coocc #look for this file in our directory


# In[76]:


a = ['a c b c b c a d d a c c a d c b a d c']

v, m = coocc.ww_matrix(a, word_tokenize, 2) #This is not the most efficient function, sorry  
m.toarray(), v
#How do you interpret this matrix? #Also, notice anything interesting?


# In[78]:


#Doing this for mobydick
mobydick = gutenberg.raw('melville-moby_dick.txt')
v, m = coocc.ww_matrix([mobydick], word_tokenize, 4) 
X = m.toarray()
X


# In[79]:


X = m.toarray()
#What is the shape of this matrix?
X.shape


# In[81]:


#Now, say we want to the word vectors for 'happy','sad','angry' again
#We need to first get the indices
happy_i = v['happy']
sad_i = v['sad']
angry_i = v['angry']


# In[82]:


happy_ww = X[happy_i,:]
sad_ww = X[sad_i,:]
angry_ww = X[angry_i,:]
np.set_printoptions(threshold=np.inf)
happy_ww  #looking at this vector for happy... 


# In[83]:


#You can look up the co-occurrence of two words within a window
#How many times does dark occur with night in a window of 4?

dark_i = v['dark']
stormy_i = v['stormy']
night_i = v['night']

X[night_i][dark_i]


# In[84]:


# Activity: See if you can identify interesting co-occurrences!


# In[ ]:


np.set_printoptions(threshold=10)


# ## PPMI Matrix
# 
# If we look at our word-word matrix, you'll see that because it is only recording the raw co-occurrences, it makes no adjustments for how certain words are just more frequent. For instance, words such as 'is' or 'the' are more likely to appear together with any other word **w** than other words. Pointwise mutual information introduces a weighting scheme to take into account co-occurence relative to two words' independent freqencies. 
# 
# Since these methods are tricks/engineering to improve results and there is no absolute truth to what is the best method. Here is one for $w$ as target word and $c$ as the context word:
# 
# 
# 
# $$PPMI_{\alpha}(w,c) = max(log_2 \frac{P(w,c)}{P(w)P_{\alpha}(c)})$$
# 
# 
# $$P_{\alpha}(c) = \frac{count(c)^{\alpha}}{\sum_c count(c)^{\alpha}}$$
# 
# 
# Notice the PPMI is taking the log ratio of the probability of co-occurrence over the probability of individual freqencies. We only consider positive value because negative probabilties (when the co-occurence is lower than expected) fluctuate. The \alpha is a measure correct for rare words with high PMI.
# 
# 
# *PPMI: Positive Pointwise Mutual Information
# 
# Levy, O., Goldberg, Y., and Dagan, I. (2015). http://www.aclweb.org/anthology/Q15-1016
# 
# 
# 

# ## Dense Vectorization
# 
# We have so far looked at vectorization methods where each element correspondence to a discrete entity such as a term from the vocabulary or a document. We have seen that this results in a lot of 0 entries. On the other hand dense vector elements are mostly non-zero, they tend to be shorter (denser) and sometimes more effective.l
# 
# Dense vectors have become more popularized lately due to deep-learning based vectors such as GloVe and Word2Vec. We will examine the truncated SVD, one dense vectorization method that is not deep-learning based but used widely. 

# **SVD**
# 
# Singular Value Decomposition (SVD) is a common method for identifying dimensions with highest variance. It is a form of dimension reduction where the algorithm identifies ways of condense as much of the information of the data in with fewer dimensions. SVD factorizes a given matrix $X$ into three matrices:
# 
# $$SVD(X) = W\Sigma C^T $$
# 
# where X is a word-word matrix, $W$ is a matrix of your new dense vectors, and $\Sigma$ is a row of singular values that represent the importance (how much variance encoded) the corresponding dimension is. Starting from the top, the first dimension encodes the most information and the following dimensions are orthogonal the the previous and contain less information down the line. A truncated SVD is the same thing but taking only $k$ top dimensions.  
# 
# See here for more information:
# https://web.stanford.edu/~jurafsky/slp3/16.pdf
# 
# 
# 
# 

# In[108]:


# You can do this super easily with sklearn!
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10) #this is your K (how many top dimensions you want to keep)


# In[109]:


denseX = svd.fit_transform(X) #n_samples, n_dims


# In[110]:


#What is the shape of this matrix?
denseX.shape


# In[111]:


happy_vector_dense = denseX[v['happy'],:]
sad_vector_dense = denseX[v['sad'],:]
angry_vector_dense = denseX[v['angry'],:]


# In[112]:


happy_vector_dense

#How does a dense vector compare to a sparse vector?


# In[113]:


happy_ww


# ## Vector Usages

# There are several ways of working with vectors and the most useful for our purposes may be similarity. We will explore this further in the next section as well.
# 

# There are several ways to measure distance between two vectors. The most widely used is **cosine similarity**. This a cosine measure of angle between two vectors. Mathematically, cosine similarity looks like this:
# 
# $$cos(\vec{v}, \vec{w}) = \frac{dot(\vec{v},\vec{w})}{norm(\vec{v})norm(\vec{w})}$$
# 
# $$dot(\vec{v},\vec{w}) = \sum_{i=0}^{n}v_iw_i$$
# $$norm(\vec{v}) = \sqrt{\sum_{i=0}^{n}v_i^2}$$
# 
# We normalize here because we want to normalize out frequency so that word similarity disregards frequency. 
# 
# Cosine similarity will range from 1 to -1. Closer to 1 means closer in direction, closer to -1 opposite in direction and something close to 0 means orthogonal.

# In[100]:


from IPython.display import Image
Image("./cosine.png")


# In[101]:


from sklearn.metrics.pairwise import cosine_similarity 


# In[114]:


# So that we can see more contrast when evaluating similarity, let's add in a rather different word: 'biscuit'

index = v['elated']
elated_vector_dense = denseX[index,:]


# In[115]:


happy_vector = happy_vector_dense.reshape(1,-1) 
sad_vector = sad_vector_dense.reshape(1,-1)
angry_vector = angry_vector_dense.reshape(1,-1)
elated_vector = elated_vector_dense.reshape(1,-1)


# In[116]:


#angle between happy and sad
cosine_similarity(happy_vector, sad_vector)


# In[117]:


cosine_similarity(angry_vector, sad_vector)


# In[118]:


cosine_similarity(happy_vector, angry_vector)


# In[119]:


cosine_similarity(happy_vector, elated_vector)


# In[ ]:


#Activity: Compare similarity of vectors of your choice!


# Using similarity as a method, we can also cluster similar vectors together. 
# This is called **clustering** and **k-means** is one popular clustering algorithm. 
# 
# K-means is an iterative algorithm that finds clusters of similar vectors by first assigning observations to their nearest means (initially randomly chosen) as its cluster and then calculating the new centroids of these clusters. 
# 
# It is called k-means because you are splitting all of your observations into k clusters by their means. 

# Let's work with a small set of words
# 
# 

# In[120]:


selection = ['green','blue','dark','yellow','bright','round','tiny','slim','square','black','thin']


# In[121]:


#iterate through all of these words to make a matrix
select_matrix = []
for word in selection:
    word_id = v[word]
    select_matrix.append(denseX[word_id,:])
select_matrix = np.array(select_matrix)
select_matrix


# In[122]:


np.array(select_matrix).shape


# In[123]:


from sklearn.cluster import KMeans


# In[124]:


kmeans = KMeans(n_clusters=3)


# In[125]:


np.set_printoptions(threshold=20)
predictions = kmeans.fit_predict(select_matrix)
predictions


# ## Visualization 

# You have probably heard of **t-sne** (is it TEA SNEA? or TAE SNAE..)!  This is "newer" dimension reduction method that emphasizes visual convenience. Sometimes PCA can produce overlapping/crowding of similar points. The con of tsne is that it is not as easily interpretable as PCA (we'll use that in part2). It's also non-deterministic -- you'll get different but similar results everytime. But we thought you should play with it here since it has been widely used in machine learning today.

# In[133]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2) #n-components = reduced dimensions


# In[134]:


#Let's make a matrix of random words
random_indices = np.random.choice(len(v), 100, replace=False)
select_matrix = X[random_indices]
lookup = {val:key for key,val in v.items()}
labels = [lookup[w] for w in random_indices]


# In[135]:


select_matrix.shape


# In[136]:


labels


# In[137]:


embed = tsne.fit_transform(select_matrix)


# In[138]:


embed


# In[139]:


random_x, random_y = zip(*embed)
fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(random_x, random_y, alpha=.8)

for _, lab in enumerate(labels):
    ax.annotate(lab, (random_x[_]+.1, random_y[_]-.05))

plt.title("random 100 embeddings")
plt.show()


# Visualizing the clustering

# In[140]:


predictions = kmeans.fit_predict(select_matrix)

first_cluster = np.array(embed[np.where(predictions == 0)])
second_cluster = np.array(embed[np.where(predictions == 1)])
third_cluster = np.array(embed[np.where(predictions == 2)])


# In[141]:


first_cluster


# In[142]:


second_cluster


# In[143]:


random_x, random_y = zip(*embed)
fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(first_cluster[:,0],first_cluster[:,1], color="blue", alpha=.5) #first cluster
ax.scatter(second_cluster[:,0],second_cluster[:,1], color="red", alpha=.5) #second cluster
ax.scatter(third_cluster[:,0],third_cluster[:,1], color="green", alpha=.5) #second cluster

for _, lab in enumerate(labels):
    ax.annotate(lab, (random_x[_]+.1, random_y[_]-.05))

plt.title("random 100 embeddings")
plt.show()


# Now, let's do this with the entire set for fun...

# In[144]:


n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters)
predictions = kmeans.fit_predict(denseX)
tsne = TSNE(n_components=2)


# In[145]:


embed = tsne.fit_transform(denseX)
random_x, random_y = zip(*embed)


# In[146]:


plt.figure(figsize=(16, 8), dpi=80)
plt.scatter(random_x, random_y, alpha=0.3)
plt.title("tsne visual of all " +str(len(v)) + " word embeddings")


# In[147]:


plt.figure(figsize=(16, 8), dpi=80)
for i in range(15):
    cluster_i = np.array(embed[np.where(predictions == i)])
    plt.scatter(cluster_i[:,0], cluster_i[:,1], c=np.random.rand(3,), alpha=.3)
plt.show()

