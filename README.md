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
> - GloVe **extends** the work of Word2Vec to capture **global** contextual information in a text corpus by calculating a global word-word co-occurrence matrix. Word2Vec, as mentined above, only captures the local context of words, during training, it only considers neighboring words to capture the context. GloVe considers the entire corpus and creates a large matrix that can capture the co-occurrence of words within the entire corpus.
> - With Glove we always use pretrained word embeddings based on very large corpus (e.g., wikipedia or twitter).
