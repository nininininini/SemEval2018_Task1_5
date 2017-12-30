ss = ['depression sucks😔', '@pepesgrandma @FreeBeacon Thanks for the early morning laughter ❤', "Wow what's up with #snap 😲📊📉⬇#cnnmoney", "I have the best girlfriend in the world 😍 #blessed"]

import emoji
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

for s in ss:
    print(tknzr.tokenize(extract_emojis(s)))



