import pandas as pd
from pathlib import Path
import re
from tqdm.auto import tqdm
import preprocessor as p
import argparse

BEGINNING_OF_TWEET_SYMBOL = '<BOT> '
END_OF_TWEET_SYMBOL = ' <EOT> '
NUM_SYMBOL = ' '

parser = argparse.ArgumentParser(description='Parse and clean tweets.')

parser.add_argument('--root_dir', type=str, default='Twitter/COVID19-Tweets-KaggleDataset/')
parser.add_argument('--output_file', type=str, default='Twitter/COVID19-Tweets-KaggleDataset-parsed_cleaned.txt')

args = parser.parse_args()

# HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

punctuations_pattern = re.compile(r'([,/$%^&*;|<>:+@#{}\[\]\\=`~()])+')

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.MENTION, p.OPT.SMILEY)


def clean_tweets(tweet: str):
    tweet = p.clean(tweet)

    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)

    # remove # from hashtags
    tweet = re.sub(r'#(\S+)', r'\1', tweet)

    #     #replace consecutive non-ASCII characters with a space except ’
    tweet = re.sub(r'(?![’])[^\x00-\x7F]+', ' ', tweet)

    #     #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    #     # consolidate repetitive punctuations
    tweet = re.sub(r'([.!?])[.!?]+', r'\1', tweet)

    # seperate punctuations from words
    tweet = re.sub(r'(\S)([.!?])', r'\1 \2', tweet)
    tweet = re.sub(r'([.!?])(\S)', r'\1 \2', tweet)

    #     # remove punctuations
    tweet = punctuations_pattern.sub(r' ', tweet)

    #     # remove numbers
    tweet = re.sub(r'\s+\d+', NUM_SYMBOL, tweet)
    tweet = re.sub(r'^\d+', NUM_SYMBOL.lstrip(), tweet)

    #     # remove repetitive white spaces
    tweet = re.sub('\s+', ' ', tweet)

    tweet = tweet.strip()

    tweet = tweet.lower()

    return tweet


with open(args.output_file, 'w') as out_file:
    for path in Path(args.root_dir).rglob('*Tweets*.CSV'):
        print(path)
        tweets = pd.read_csv(path)
        tweets = tweets[['text', 'lang']]

        for index, row in tqdm(tweets.iterrows()):

            if row['lang'] == 'en':
                tweet = row['text']
                tweet = clean_tweets(tweet)
                out_file.write(BEGINNING_OF_TWEET_SYMBOL + tweet + END_OF_TWEET_SYMBOL)
