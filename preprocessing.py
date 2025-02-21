import re
import demoji
import nltk
import concurrent.futures
import json
import malaya
import jieba
from num2words import num2words
from googletrans import Translator
from langdetect import detect
import langdetect
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

    def lemmatize_mixed_text(self, text):
        words = word_tokenize(text)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            lemmatized_words = list(executor.map(self.lemmatize_word, words))
        return " ".join(lemmatized_words)

    def lemmatize_word(self, word):
        try:
            lang = detect(word)
        except:
            lang = "en"
        
        if lang == "en":
            return self.lemmatize_english(word)
        elif lang == "ms":
            return self.lemmatize_malay(word)
        elif lang in ["zh-cn", "zh-tw"]:
            return self.segment_chinese(word)
        return word

class Preprocessing():
    def __init__(self,additional_fun=None):
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
        self.additional_fun = additional_fun

    def stopwords_removal(self, string):
        return " ".join(filter(lambda x: x not in self.stopwords, string.split()))
        
    def digit2word(self, string):
        return re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang="en").replace(" ", "_"), string)
    
    def demoji_text(self, string):
        emoji_dict = demoji.findall(string)
        if not emoji_dict:
            return string
        return re.sub(
            "|".join(map(re.escape, emoji_dict)),
            lambda match: emoji_dict[match.group(0)].replace(" ", "_"),
            string
        )

    def remove_email_url(self, string):
        return re.sub(r"https?://\S+|\S+@\S+", "", string)

    def remove_whitespace(self,string):
        string = re.sub(r"\n(?!\n)", " ", string)
        string = re.sub(r"\n+", "\n", string)
        string = re.sub(r" +", " ", string).strip()
        string = re.sub(r"\n+", " ", string)

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
    
    async def translate_if_needed(self, string):
        try:
            detected_lang = detect(string)
            if detected_lang in ['id', 'zh']:
                translated_text = await self.translator.translate(string, src=detected_lang, dest='en')
                return translated_text.text
        except langdetect.lang_detect_exception.LangDetectException:
            return string 
        return string
    
    async def preprocessing_pipeline(self, text):
        text = str(text)
        text = self.remove_email_url(text)
        text = self.separate_chinese_english(text)
        text = self.expand_slang(text)
        text = self.handle_negation(text)
        text = self.digit2word(text)
        text = self.remove_punctuation_exception(text)
        text = self.demoji_text(text)
        text = self.remove_whitespace(text)
        text = await self.translate_if_needed(text)
        text = self.multilingualLemmatize(text)
        text = self.stopwords_removal(text)

        if self.additional_fun:
            text = self.additional_fun(text)

        return text



