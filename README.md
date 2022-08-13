<h3 align="center">NLP: SENTIMENT ANALYSIS</h3>

- - -
> Reviews on a product, service, movie or texts about persons or events (political, social, etc.), are very important to get a clear picture of what the end users/general public think, namely, to understand what the reasons (key aspects/features) are for being satisfied or not with the purchase of a product or service, the reasons for liking or disliking a given movie or person, and so on. 
>
> **`Sentiment analysis` can help us gather insightful information regarding reviews/texts by deciphering what people like/dislike, what they want and what their major concerns are**.<br>

>There are mainly two approaches to extract the sentiment from given reviews/texts and classify the result as positive or negative: 
>- Lexicon Based Approach 
>- Machine Learning Approach  

→ In the **notebooks** of this **repo** we will address the **`machine learning approach`**

> **Machines simply cannot process text data in raw form**. They need us to **break down the text into a numerical format that’s easily readable by the machine**. This is where the **techniques/models described below come into play**.

**1 - `Bag of Words` (BoW) and `TF-IDF`**

> - BoW and TF-IDF are techniques that help us convert text sentences into numeric vectors.
> - BoW just creates a set of vectors containing the count of word occurrences in the document (reviews), while the TF-IDF technique creates a normalized count where each word count is divided by the number of documents this word appears in, being so, it captures both the relevance and frequency of a word in a document.
> - For both, BoW and TF-IDF, each word is captured in a **standalone** manner, thus the **local context (surounding words) in which it occurs is not captured**, therefore they both **don't consider the semantic relation between words**.

→ **Context** can be seen in **two different perspectives that we will address later on**.

**2 - `Word2Vec` and `Glove`**
> - Word2vec is a two-layer neural net that processes text by vectorizing words. Its input is a text corpus and its output is a set of vectors: feature vectors that represent words in that corpus.
> - Word2Vec considers **neighboring words to capture the local context of a word**, while at the same time reducing the size of the data. 
> - When using Word2Vec, if we think we have enough data, one can go for a custom vectorization as it will be very specific to the context the corpus has or one can use pretrained word embeddings trained in a very large corpus (Gopgle News)
> - Word2Vec and Glove **differ in the way they are trained**. GloVe **extends** the work of Word2Vec to **capture global contextual information** in a text corpus by calculating a global word-word co-occurrence matrix. Word2Vec, as mentined above, **only captures the local context of words** during training, it only considers neighboring words to capture the context. GloVe considers the entire corpus and creates a large matrix that can capture the co-occurrence of words within the entire corpus.
> - Pretrained models for both these embeddings are readily available and are easy to incorporate into python code.
- - - 

**3 - `BERT`**

> - While Word2vec and Glove word embeddings are **context independent**, in the sense that **these models output just one vector (embedding) for each word, combining all the different senses of the word into one vector**, BERT can **generate different word embeddings for the same word** that **captures** the **context** of the **word** - that is **its position in a sentence**.
> - Word2vec and Glove **do not consider word order in their training**, while BERT **takes into account word order (BERT uses Transformer - an attention based model with positional encodings to represent word positions)**.
>     - For instance, for the following sentence "He went to the prison **cell** with his **cell** phone to extract blood **cell** samples from inmates", BERT would **generate three different vectors for cell**. The first 'cell' (**prison cell case**)  would be closer to words like incarceration, crime etc., whereas the second 'cell' (**phone case**) would be closer to words like iPhone, android, etc., the third 'cell' (**blood cell case**) would be closer to words such as platelets, hemoglobin, fluid, etc.
> - A **practical implication of the difference mentioned above** is that we **can use Word2vec and Glove vectors directly for downstream tasks**, all we need is the vectors for the words, **there is no need for the model itself that was used to train these vectors**. However, in the case of BERT, since it is context dependent, **we need the model that was used to train the vectors even after training, since the models generate the vectors for a word based on context**.
- - - 

**4 - `Doc2Vec`**

> - Doc2Vec is an **extension** of Word2Vec that **is applied to a document/review as a whole instead of individual words**. This model **aims to create a numerical representation of a document/review rather than a word** (Le & Mikolov, 2014). Doc2Vec **operates on the logic that the meaning of a word also depends on the document that it occurs in**.
> - The **tags are just keys into the doc-vectors collection**, they **have no semantic meaning**.
- - - 

### `Sentiment Analysis of the Restaurant Reviews from YELP Dataset`

&rarr; The two datasets (**review** and **business**) that we need from **YELP** dataset can be found [here](https://www.yelp.com/dataset)

&rarr; You can learn how to **create a user-managed notebook instance in Google Cloud Platform** [here](https://cloud.google.com/vertex-ai/docs/workbench/user-managed/create-new)

- **Tasks performed in the notebooks - Key Points**:

> **1** - **Preprocessing** (a whole notebook for this task)

> **2** - **BoW** and **TF-IDF**:
>
> 2.1 - Firstly, **read the cleaned dataset stored in the bucket after the preprocesssing phase**
> 
> 2.2 - Do the **train/test split keeping the two columns of interest and preserve the same proportions of examples in each class as observed in the original dataset**
>
> 2.3 - **Establishing benchmarks**: **Logistic Regression** and **SVM models** with the **default parameters** that will **serve** as **baseline models**. **Evaluation**.
>
> - **Note**: Using **GridSearchCV** I found out for both **BoW** and **TF-IDF** which **ngram_range** ((1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3)) **performed better** with the **chosen classifiers**: **Logistic Regression** and **SVM**
>
> 2.4 - **Logistic Regression and SVM classifiers** with **hyperparameter tuning** (using **GridSeachCV**). **Evaluation**.

> **3** - **Word2Vec**:
>
> - **Note**: **Although the dataset is not very large**, I used Word2Vec **without using pretrained word embeddings** because I thought I **had enough data** to go for a **custom vectorization** as it **would be very specific to the context the corpus had**.
>
> 3.1 - **Learning word embeddings on training data**
>
> 3.2 - **Create a function to get the vectors that will feed the classifiers**
>
> 3.3 - **Establishing benchmarks**: **Logistic Regression** and **SVM classifier models** with **default parameters** that will **serve** as **baseline models**:
>
> 3.3.1 - **Evaluation using Logistic Regression**
>
> 3.3.2 - **Evaluation using SVM**
>
> 3.4 - **Hyperparameter tuning and evaluation**:
>
> 3.4.1 - **Hyperparameter tunning** using **GridSearchCV** for **Logistic Regression**. **Evaluation**.
>
> 3.4.2 - **Hyperparemeter tunning** using **GridSearchCV** for **SVM**. **Evaluation**

> **4** - **Doc2Vec**:

> **5** - **BERT** using **PyTorch** framework (a **whole Colab** notebook). For an **interactive preview** &rarr; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josepaulosa/NLP_Sentiment_Analysis/blob/main/BERT.ipynb)
>
> 5.1 - **Preprocessing**
> 
> 5.2 - Install **`transformers`**
> 
> 5.3 - Define the **pre-trained** model we are going to use: **`bert-base-uncased`**
> 
> 5.4 - Load the **BERT Tokenizer**
> 
> 5.5 - Decide on a **maximum sentence length**
> 
> 5.6 - **Split** the **dataset** into **train** (70 %), **validation** (15%) and **test** (15%) sets
> 
> 5.7 - **Convert to list each of the sets**
> 
> 5.8 - Create a **sequence of token ids** (input ids) for **each review**
> 
> 5.9 - **Padding** and **truncating**: **pad** and **truncate** our **sequences** so that they **all have the same length**
> 
> 5.10 - Create **attention masks**
> 
> 5.11 - **Convert all lists of inputs ids, labels and attention masks into torch tensors**
> 
> 5.12 - Create a **`DataLoader`** to **load** our **datasets**
> 
> 5.13 - **Build a sentiment classifier**: **`BertForSequenceClassification`**
> 
> 5.14 - **Instantiate** our model
> 
> 5.15 - **Move** our **model** to the **GPU**
> 
> 5.16 - **Optimizer, learning rate scheduler, loss function and number of epochs**
>
> **Notes**: 
>
> - To **fine-tune** our Bert classifier we need to create an **`optimizer`**. Considering the **original paper** on BERT, we will use **`AdamW`** optimizer because it **implements gradient bias correction** as well as **weight decay**. We will also use **linear scheduler with no warmup steps**.
>
> - The **authors** have some **recommendations** for **fine-tuning**:
>
>     - Batch size: 16, 32
>     - Learning rate (Adam): 5e-5, 3e-5, 2e-5
>     - Number of epochs: 2, 3 or 4
>
> We chose **16**, **3e-5** and **2**, respectively.
>
> 5.17 - **Training** and **validation**
> 
> - **Note**: We can **store** the **training** and **validation loss** and **accuracy values** and then **plot** or **make a table** to **measure the performance** on **both** the **train** and **validation** sets **after the completion of each epoch** so we can see **more clearly** the **progress** of the **training loop**.
>
> 5.18 - **Predicting on test set**
>
> 5.19 - **Converting predictions to tensor**
> 
> 5.20 - **Applying softmax on predictions**
> 
> 5.21 - **Conversion to numpy arrays**
> 
> 5.22 - **Heatmap of confusion matrix for test data**
> 
> 5.23 - **Classification report**
>
> 5.24 - **Conclusions**

>
>
>
>



















