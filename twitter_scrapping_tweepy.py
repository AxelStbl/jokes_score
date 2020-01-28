import tweepy
import json

with open('private.json') as json_file:
    data = json.load(json_file)
    consumer_key, consumer_secret, access_key, access_secret = data[
                                                                   'consumer_key'], \
                                                               data[
                                                                   'consumer_secret'], \
                                                               data[
                                                                   'access_key'], \
                                                               data[
                                                                   'access_secret']
    print(consumer_key, consumer_secret, access_key, access_secret)


# Function to extract tweets
def get_tweets(username):
    # Authorization to consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    # Access to user's access key and access secret
    auth.set_access_token(access_key, access_secret)

    # Calling api
    api = tweepy.API(auth)

    tweets_store = []
    for status in tweepy.Cursor(api.user_timeline,
                                screen_name=username,
                                tweet_mode="extended").items():
        tweets_store.append(status.full_text)
        print(status.full_text)
    return tweets_store


# Driver code
if __name__ == '__main__':
    # Here goes the twitter handle for the user
    # whose tweets are to be extracted.
    print(len(get_tweets("@Shen_the_Bird")))
