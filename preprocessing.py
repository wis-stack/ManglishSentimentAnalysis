import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

english_stopwords = set(stopwords.words('english'))

with open("stopwords/cn_stopwords.txt", "r", encoding="utf-8") as file:
    chinese_stopwords = {line.strip() for line in file}

with open("stopwords/manglish_stopwords.txt", "r", encoding="utf-8") as file1:
    manglish_stopwords = {line.strip() for line in file1}

with open("stopwords/ms_stopwords.txt", "r", encoding="utf-8") as file2:
    malay_stopwords = {line.strip() for line in file2}

outputstopwords = english_stopwords | chinese_stopwords | manglish_stopwords | malay_stopwords

print(outputstopwords)