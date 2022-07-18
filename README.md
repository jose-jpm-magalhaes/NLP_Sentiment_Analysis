<h3 align="center">NLP: SENTIMENT ANALYSIS</h3>

- - -
> Reviews on a product, service, movie or texts about persons or events (political, social, etc.), are very important to get a clear picture of what the end users/general public think, namely, to understand what the reasons (key aspects/features) are for being satisfied or not with the purchase of a product or service, the reasons for liking or disliking a given movie or person, and so on. 
>
> **Sentiment analysis can help us gather insightful information regarding reviews/texts by deciphering what people like/dislike, what they want and what their major concerns are**.<br>

>There are mainly two approaches to extract the sentiment from given reviews/texts and classify the result as positive or negative: 
>- Lexicon Based Approach 
>- Machine Learning Approach  

→ In the **notebooks** of this **repo** we will address the **machine learning approach**

> **Machines simply cannot process text data in raw form**. They need us to **break down the text into a numerical format that’s easily readable by the machine**. This is where the **techniques/models described below come into play**.
- - - 
**1 - Bag of Words (BoW) and TF-IDF**

> - BoW and TF-IDF are techniques that help us convert text sentences into numeric vectors.
> - BoW just creates a set of vectors containing the count of word occurrences in the document (reviews), while the TF-IDF technique creates a normalized count where each word count is divided by the number of documents this word appears in, being so, it captures both the relevance and frequency of a word in a document.
> - For both, BoW and TF-IDF, each word is captured in a **standalone** manner, thus the **local context (surounding words) in which it occurs is not captured**, therefore they both **don't consider the semantic relation between words**.

→ **Context** can be seen in **two different perspectives that we will address later on**.

**2 - Word2Vec and Glove**
> - Word2vec is a two-layer neural net that processes text by vectorizing words. Its input is a text corpus and its output is a set of vectors: feature vectors that represent words in that corpus.
> - Word2Vec considers **neighboring words to capture the local context of a word**, while at the same time reducing the size of the data. 
> - When using Word2Vec, if we think we have enough data, one can go for a custom vectorization as it will be very specific to the context the corpus has or one can use pretrained word embeddings trained in a very large corpus (Gopgle News)
> - Word2Vec and Glove **differ in the way they are trained**. GloVe **extends** the work of Word2Vec to **capture global contextual information** in a text corpus by calculating a global word-word co-occurrence matrix. Word2Vec, as mentined above, **only captures the local context of words** during training, it only considers neighboring words to capture the context. GloVe considers the entire corpus and creates a large matrix that can capture the co-occurrence of words within the entire corpus.
> - Pretrained models for both these embeddings are readily available and are easy to incorporate into python code.
- - - 

**3 - BERT**

> - While Word2vec and Glove word embeddings are **context independent**, in the sense that **these models output just one vector (embedding) for each word, combining all the different senses of the word into one vector**, BERT can **generate different word embeddings for the same word** that **captures** the **context** of the **word** - that is **its position in a sentence**.
> - Word2vec and Glove **do not consider word order in their training**, while BERT **takes into account word order (BERT uses Transformer - an attention based model with positional encodings to represent word positions)**.
>     - For instance, for the following sentence "He went to the prison **cell** with his **cell** phone to extract blood **cell** samples from inmates", BERT would **generate three different vectors for cell**. The first 'cell' (**prison cell case**)  would be closer to words like incarceration, crime etc., whereas the second 'cell' (**phone case**) would be closer to words like iPhone, android, etc., the third 'cell' (**blood cell case**) would be closer to words such as platelets, hemoglobin, fluid, etc.
> - A **practical implication of the difference mentioned above** is that we **can use Word2vec and Glove vectors directly for downstream tasks**, all we need is the vectors for the words, **there is no need for the model itself that was used to train these vectors**. However, in the case of BERT, since it is context dependent, **we need the model that was used to train the vectors even after training, since the models generate the vectors for a word based on context**.
