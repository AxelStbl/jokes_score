# Jokes Regressor (Funny-Meter)

### Contributors:

- Axel Strubel
- Avi Barazani
- Yaniv Ben-Malka

### Summary

Using machine learning tools, we have built a supervised nlp-based jokes regressor, that gets an English joke as an input, creates various features that try to capture the sense of the joke, and predicts how funny it is, in a scale 0 to 100.

### Dataset

A reddit jokes dataset, that contains more than 180,000 jokes and their score (votes). 

Our future intention is to use the scrap tweets from a pre-made list of 'funny' pages, and use the tweets to further train & test the model, in addition to some more twitter-based feature engineering.

### Files

**Notebooks** 
  - EDA
  - Feature Engineering
  - Feature Processing (Dimensionality Reduction)
  - Modeling
####
**Joke_Predictor** 
  - predict_joke.py (Run the entire pipline for predicting an input joke score)
####
**Data** 
  - reddit_jokes.json - Dataset currently in use
####
**Twitter_API (Not in use)** 
- Get_Tweets.py - Tweets scraper
- tweets.csv - A dataset of scraped tweets

 
### Project description (Extended version)

docs.google.com/document/d/1f_8DQBT13SHORA22K4gOfqOf_l_9iXPtuVgdrXE55Cw
