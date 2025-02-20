import re
import demoji
import nltk
import spacy
import json
import malaya
import jieba
from num2words import num2words
from googletrans import Translator
from langdetect import detect
from nltk.tokenize import word_tokenize

demoji.download_codes()
nltk.download('punkt_tab')

def load_spacy_model(model_name="en_core_web_sm"):
    try:        
        import spacy
        return spacy.load(model_name)
    except OSError:
        print(f"⚠️ SpaCy model '{model_name}' not found. Downloading...")
        import spacy.cli
        spacy.cli.download(model_name)
        return spacy.load(model_name)
    
class MultilingualLemmatizer():

    def __init__(self):
        self.nlp_en = load_spacy_model("en_core_web_sm")
        self.nlp_ms = malaya.stem.sastrawi()

    def lemmatize_english(self,text):
        doc = self.nlp_en(text)
        return " ".join([token.lemma_ for token in doc])

    def lemmatize_malay(self, text):
        doc = self.nlp_ms(text)
        return " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])

    def segment_chinese(self, text):
        return " ".join(jieba.cut(text))

    def lemmatize_mixed_text(self,text):
        words = word_tokenize(text)
        lemmatized_words = []

        for word in words:
            try:
                lang = detect(word)
            except:
                lang = "en"

            if lang == "en":
                lemmatized_words.append(self.lemmatize_english(word))
            elif lang == "ms":
                lemmatized_words.append(self.lemmatize_malay(word))
            elif lang == "zh-cn" or lang == "zh-tw":
                lemmatized_words.append(self.segment_chinese(word))
            else:
                lemmatized_words.append(word) 

        return " ".join(lemmatized_words)

class Preprocessing():
    def __init__(self,tokenization_fun):
        with open("stopwords/cn_stopwords.txt", "r", encoding="utf-8") as file:
            chinese_stopwords = {line.strip() for line in file}

        with open("stopwords/manglish_stopwords.txt", "r", encoding="utf-8") as file1:
            manglish_stopwords = {line.strip() for line in file1}

        with open("stopwords/ms_stopwords.txt", "r", encoding="utf-8") as file2:
            malay_stopwords = {line.strip() for line in file2}
        
        with open("stopwords/en_stopwords.txt", "r", encoding="utf-8") as file3:
            english_stopwords = {line.strip() for line in file3}
        
        with open("contractions/contractions.json", "r", encoding="utf-8") as file4:
            self.contractions = json.load(file4)
        
        self.stopwords = set(english_stopwords | chinese_stopwords | manglish_stopwords | malay_stopwords)
        self.lemmatizer = MultilingualLemmatizer()
        self.translator = Translator()
        self.tokenization_fun = tokenization_fun

    def stopwords_removal(self,string):
        words = string.split()
        return " ".join([x for x in words if x not in self.stopwords]) 
    
    def digit2word(self,string):
        def changeDigit(match):
            matched_string = match.group(0)
            return num2words(matched_string, lang='en').replace(" ", "_")
        
        if any(char.isdigit() for char in string): 
            string = re.sub(r"[0-9]+", changeDigit, string)

        return string
    
    def demoji_text(self,string):
        emoji_dict = demoji.findall(string)

        if not emoji_dict:
            return string

        emoji_pattern = re.compile(
            "|".join(re.escape(emoji) for emoji in emoji_dict.keys())
        )

        def replace_emoji(match):
            emoji = match.group(0)
            emoji_text = emoji_dict.get(emoji, emoji)
            return emoji_text.replace(" ", "_")

        return emoji_pattern.sub(replace_emoji, string)
    
    def remove_email_url(self,string):
        string = re.sub(r"http\S+", "", string)
        string = re.sub(r"https\S+", "", string)
        string = re.sub(r"\S+@\S+", "", string)

        return string
    
    def remove_whitespace(self,string):
        string = re.sub(r"\n(?!\n)", " ", string)
        string = re.sub(r"\n+", "\n", string)
        string = re.sub(r" +", " ", string).strip()

        return string

    def separate_chinese_english(self, text):
        text = re.sub(r'([\u4e00-\u9fff])([^\u4e00-\u9fff])', r'\1 \2', text)
        text = re.sub(r'([^\u4e00-\u9fff])([\u4e00-\u9fff])', r'\1 \2', text)
        text = re.sub(r'\s+([,.!?])', r'\1', text)  
        text = re.sub(r'([,.!?])\s+', r'\1 ', text)
        return text.strip()

    
    def expand_slang(self, string):
        return " ".join([self.contractions.get(word, word) for word in string.split()])

    def handle_negation(self, string):
        string = re.sub(r"\b(not|never|no|不|tak|bukan)\s+(\w+)", r"\1_\2", string)
        return string
    
    def multilingualLemmatize(self,string):
        return self.lemmatizer.lemmatize_mixed_text(string)
    def remove_punctuation_exception(self, string):
        return re.sub(r"[^\w\s!?]", "", string)
    
    def translate_if_needed(self,string):
        detected_lang = self.translator.detect(string)
        if detected_lang == 'id': 
            translated_text = self.translator.translate(string, src='id', dest='en').text
            return translated_text
        elif detected_lang == 'zh':
            translated_text = self.translator.translate(string, src='zh', dest='en').text
            return translated_text
        else:
            return string
    
    def preprocessing_pipeline(self, text):
        text = text.lower()
        text = self.separate_chinese_english(text)
        text = self.expand_slang(text)
        text = self.handle_negation(text)
        text = self.digit2word(text)
        text = self.remove_email_url(text)
        text = self.remove_punctuation_exception(text)
        text = self.remove_whitespace(text)
        text = self.demoji_text(text)
        text = self.translate_if_needed(text)
        text = self.multilingualLemmatize(text)
        text = self.stopwords_removal(text)
        tokens = self.tokenization_fun(text)
        
        return tokens


