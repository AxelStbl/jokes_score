
Achievements summary

Buliding a supervised nlp-based jokes regressor, that gets a joke in English as an input, creates various features that try to capture the sense of the joke, and predicts how funny it is, in a scale 0 to 1.

Working Process & Results 

Decision on which dataset to use

Our initial plan was to scrape joke tweets from twitter, and analyze them according to twitter popularity metrics. 
Due to some complications that will be discussed in Section # 3 (Challenges), and limited schedule, we have decided to enhance the core process of building a POC model, by using an already-prepared Reddit jokes dataset, that includes more than 100k jokes and their score based on viewers votes.


Literature review

As an initial phase, we have read multiple articles about Humor theory. 
Articles referred to the general structure of jokes,main joke types and the elements that makes jokes funny. 
In addition, we have read multiple articles about attempts to build unsupervised models that classifies texts to funny/not funny. 
These articles were a great source for inspiration regarding methods that can be used in analyzing humorous texts.
Article list can be found in the end of the file (Literature Section).
We can say that  this phase contributed a lot to the process of feature engineering and post-model analysis.






Exploratory Data Analysis 
We can see that we have 86% of our score target variable comprised between 1 and 101 and up to around 4000 for 99% of our scores.
The number of character counts in jokes seems to follow a normal central law rightly skewed.

Some of the most frequent word across all jokes are “one”, “see”, “call” and in the top 20% of jokes its “say”, “one” and “man”.
We have created a panda table showing statistics like the number of capitals, number of exclamation mars, mean of word length per Joke and we used it to display a correlation matrix of all this numerical features created .














Pre-Processing

Below are the steps taken in the feature engineering and data pre-processing, in their actual order:

Removing jokes with 0 score

The dataset includes around 50,000 jokes (out of 180,000). with a score of 0.
When running our baseline model, having these jokes in the data severely affected the results and caused almost all of the jokes to receive a score of 0.  
Since we can’t be sure that the reason that these jokes received a 0 score is because they’re bad, or just originates missing data or lack of proper presentation for voting, we have decided to remove them and work only with jokes with positive score. 
132,992 samples remained on the dataset.
 

Filter jokes by length

Following the EDA, our dataset included only 2 columns: joke and score.
In order to remove outliers and increase computational efficiency, we have decided to check the distribution of jokes length. 
Maximum joke length was above 7,000 words, and only 25% of the jokes were above 40 words, which is the inflection point in the plot below.  
We have decided to remove from dataset all jokes that are longer than 40 words. 98,563 jokes were left on the dataset.





Feature engineering

Building space-agnostic features
Building features that are originated from the joke itself and not from the entire space, and therefore can be done before splitting to train/test set:

Count Ambiguous words
Common humor method is to use ambiguous words - Using a word that has multiple unrelated meanings. (Example: Break something / Take a break). 
This feature is a counter of occurrences of ambiguous words within the joke. T
his was achieved by scraping a list of ambiguous words from the web, and count occurrences of any of them in the joke.

Incongruity (Repetitions)
Some jokes are based on Repetitiveness, which is reflected by usage of words that have close meaning multiple times. 
This feature gets the maximum similarity between two words within a joke. Similarity was measured based on the pre-trained embeddings of the brown corpus.

Antonyms
Check whether a word and its opposite appear in the a joke.
Using wordnet antonym feature, and iterate over all word pairs in each joke.

Longest word
Length of the longest word in a joke.

Special chars
Total amount of special characters in a joke.

Mean Characters Per Word
Determines the mean number of characters per word.
Done using featuretools package.

Polarity Score
Calculates the polarity of a text on a scale from -1 (negative) to 1 (positive) - whether text has negative or positive sentiment.
Done using featuretools package.

Punctuation Count
Determines number of punctuation characters in a string.
Done using featuretools package.

Stopword Count
Determines number of stopwords in a string.
Done using featuretools package.

Title Word Count
Determines the number of title words in a string.
Done using featuretools package.

Upper Case Count
Calculates the number of uppercase letters in text.
Done using featuretools package.


Sentence embeddings

Embeddings are n-dimensional representations of a word. 
In order to get the embedding of a sentence, an average is done over all words. We have decided to add the entire sentence embedding vector to our dataset, and later reduce its dimensionality using feature selection methods.
Following some experiments that will be discussed in ‘Challenges’, we have decided to use featuretools package to train 512-dimensional embeddings for our jokes dataset.

Objects
We would like to tell on which objects the joke refer (People, places etc.). 
Using Spacy tokenization, the amounts of occurrences of each object type were added as features.
We would like to tell the structure of the joke, and whether it contains more nouns, more verbs, more punctuation etc. 
Using Spacy POS methods, the amounts of occurrences of each POS element were added as features. 

Total words
Amount of tokens in a joke

Total chars
Amount of characters in a joke

Additional features we planned to develop but were not completed due to limited schedule:

Does joke contains a pun (Word that sounds like another word)
Longest rhyme chain
Alliteration - Find similar letters in consecutive words
Profanity - Identify if joke is ‘dirty’ or not



Text cleaning

The last feature set we have created is the TF/IDF vectors. Since these features are based on the entire vocabulary, we’ll clean the text first. Then, in order to avoid leakage, we would split the dataset into train and test set.

The following text cleaning actions were taken:

Remove non-english punctuation
Replace escape characters
Remove multi-spaces
Remove special characters
Remove all single characters
Remove prefixed 'b' (Binary)
Convert to lowercase
Lemmatization

Train/Test split

Split to train/test set was done randomly. Since dataset is relatively large, a test set of only 20% was used.

TF/IDF Vectorizer features

We used 1000-tfidf vectorizer, for the 1000 most common 1-2 grams. 
Add the 1000-dimensional vector to the dataframe. 
We have decided to add the entire sentence embedding vector to our dataset, and later reduce its dimensionality using feature selection methods.

Modeling

As part of our initial research, it was found that there are 5 main humor types: Human-centric vocabulary, Negation Professional communities, Negative orientation and Human “weakness”. 
Accordingly, we have tried to use K-means unsupervised clustering for divding the jokes into 5 clusters, that hopfully each one will represent each humor type.
Our plan was to train a separate model for each cluster. 
Unfortumately, K-means did not perform well (large distances between points - inertia_), and due to limited schedule we have decided to skip this phase at the moment.
We tried then Linear Regression with and without normalization. It performed slightly better without Normalization.
Then we tried several models with some grid search in order to improve the parameters selection. The following models were tested : ElasticNet, Random Forest Regressor and XGboost regressor. The XGboost regressor displayed the best accuracy among them.
We tried also a simple neural network with two hidden layers with “relu” and “linear” activation function.


Post - Model analysis (TBD)

Humor anchors
Find what are the ‘Humor anchors’ of each joke.  
A humor anchor is a meaningful, complete, minimal set of word spans in a sentence that potentially enable one of the latent structures of Section 4 to occur.
Find which humor anchors are most common among funny jokes

Commonalities
Check what are the closest words to the sentence embeddings of the most funny jokes, and see if we can spot some patterns.

General
Feature influence analysis with SHAP/Lime
Feature importance





Did you model succeed? Failed? Describe your results in both a technical and business strategy approach.

 Challenges

Scraping twitter

We started to scrape twitter API and it was a big challenge since we needed a developer account and it took time to obtain one. 
In addition, twitter api limits the data we can get from one particular account. We decided to focus on the text specifically and switch to Reddit dataset where we have already more than 100 000 rows.

Embeddings

Initially, we used gensim google-news-negative 300 dimensional pre-trained embeddings. However, the initial results of the models were not good, and we suspected that news embeddings may not proprely represent the relations between words in a joke. 
Then, we tried to use twitter-based embeddings, under the assumption that they will represent better slang and informal speech. However, a large ratio of the common vocabulary was missing. 
In continuation, we have decided to try to train our own embeddings, using an embedding layer in a Keras neural network. This did not go well, as the results did not make sense. When underfitted, most of the words within the same joke received similar embeddings, and when overfit all of the words in the vocabulary received embeddings very close to 1.
Our final solution for this issue was to use deep feature synthesis (DFS) with featuretools package, for generating embeddings based on our jokes dataset.

Runtime

We have used a large dataset which took a few hours to process after each feature revision, both in our local machines or in Google Colab’s GPU. 
Therefore, we have usually used only part of the dataset, which caused the models to perform poorly, without us knowing whether its because we need to add more features or because we need to use the entire dataset. 
Only as a last resort, we decided to use Cnvrg.io most powerful server (gpuxl), which processed the entire dataset in a reasonable time and allowed us to perform proper testing.

 Things we would have done differently

Workflow

Our baseline model from CP# 2 did not perform well. 
Following our literature research, we have made a list of features we would like to implement, and put our best efforts on adding as much of them as possible. 
Due to the effort given to this phase which took some time, and due to the fact that we used a large dataset which took a few hours to fully process after each feature revision. it came out that only in a very late stage we were able to run the model on the entire dataset with all features and see its actual performance. 
In earlier stages, the models did not perform well, but we weren’t sure if it’s the lack of features or due to the fact the we were not running it on the entire dataset. 
This is not a desired situation. We should have given this a prior thought and figure out a way to get a valid sense of model performance in an earlier stage.

Dataset choice

We could have seen quicker what dataset we should use and maybe start with one that is already ready. In a second time we could have tried to do some scrapping.

 Literature

https://www.aclweb.org/anthology/C18-1157.pdf
https://nlp.stanford.edu/courses/cs224n/2013/reports/cai.pdf
https://www.cs.cmu.edu/~alavie/papers/D15-1284.pdf
http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp05b.pdf
https://arxiv.org/pdf/1512.04407.pdf
https://www.theatlantic.com/magazine/archive/2018/03/funny-how/550910/#7
https://owencurtis.com/portfolio/reddit-com/
https://www.itc.tech/an-algorithm-that-predicts-how-funny-a-text-is-yup-its-a-real-thing/
https://www.cs.upc.edu/~horacio/ahlt/DeepLearning01.pdf

