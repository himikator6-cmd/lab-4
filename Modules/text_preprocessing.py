import json
import re
import nltk
import pymorphy2




def clean_html(html_text):
    cleaned_text = re.sub(r'<[^>]+>', '', html_text)
    cleaned_text = re.sub(r'&[^;]+;', '', cleaned_text)
    return cleaned_text


def text_lower(text):
    lower_text = text.lower()
    return lower_text


def clean_space(space_text):
    cleaned_text = re.sub(r'\s+', ' ', space_text).strip()
    return cleaned_text


def cleaner(text):
    text1 = clean_html(text)
    text2 = text_lower(text1)
    text3 = clean_space(text2)
    return text3




def clean_abbreviations(text):
    cleaned_text = re.sub(r'\sг\.', ' год', text)
    cleaned_text = re.sub(r'\sс\.', ' страница', cleaned_text)
    cleaned_text = re.sub(r'\sв\.', ' век', cleaned_text)
    cleaned_text = re.sub(r'\sавт\.', ' автор', cleaned_text)
    cleaned_text = re.sub(r'\sзам\.', ' заместитель', cleaned_text)
    cleaned_text = re.sub(r'\sчел\.', ' человек', cleaned_text)
    cleaned_text = re.sub(r'\sсм\.', ' смотреть', cleaned_text)
    cleaned_text = re.sub(r'\sгг\.', ' годы', cleaned_text)
    cleaned_text = re.sub(r'\sпр\.', ' прочие', cleaned_text)
    cleaned_text = re.sub(r'\sдр\.', ' другие', cleaned_text)
    cleaned_text = re.sub(r'\sнапр\.', ' например', cleaned_text)
    cleaned_text = re.sub(r'\sт\.е\.', ' то есть', cleaned_text)
    cleaned_text = re.sub(r'\sт\.д\.', ' так далее', cleaned_text)
    cleaned_text = re.sub(r'\sт\.п\.', ' тому подобное', cleaned_text)
    cleaned_text = re.sub(r'\sт\.н\.', ' так называемый', cleaned_text)
    cleaned_text = re.sub(r'\sт\.к\.', ' так как', cleaned_text)
    cleaned_text = re.sub(r'\sт\.о\.', ' таким образом', cleaned_text)
    cleaned_text = re.sub(r'\sн\.э\.', ' нашей эры', cleaned_text)
    cleaned_text = re.sub(r'\sед\.ч\.', ' единственное число', cleaned_text)
    cleaned_text = re.sub(r'\sмн\.ч\.', ' множественное число', cleaned_text)
    cleaned_text = re.sub(r'\sв\sт\.ч\.', ' в том числе', cleaned_text)
    cleaned_text = re.sub(r'\sизд-во', ' издательство', cleaned_text)
    cleaned_text = re.sub(r'\sгос-во', ' государство', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def clean_punctuation(text):
    cleaned_text = re.sub(r'[,;!.:?%]\s', ' ', text)
    cleaned_text = re.sub(r'\.{3}', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s[-—=+*/]\s', ' ', cleaned_text)
    cleaned_text = re.sub(r'[,;!.:-=+*/?%]$', '', cleaned_text)
    cleaned_text = re.sub(r'["\'«»„“”‘’\(\)\[\]\{\}\<\>\|]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def clean_stopwords(text):
    words = text.split(' ')
    russian_stopwords = nltk.corpus.stopwords.words("russian")
    cleaned_words = [word for word in words if word not in russian_stopwords]
    cleaned_text = ' '.join(cleaned_words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def unification(text):
    unificated_text = re.sub(r'\s[a-z]+\s', ' [NAME] ', text)
    unificated_text = re.sub(r'\s[a-z]+\s', ' [NAME] ', unificated_text)
    unificated_text = re.sub(r'-[a-z]+', '-[NAME]', unificated_text)
    unificated_text = re.sub(r'[a-z]+-', '[NAME]-', unificated_text)
    unificated_text = re.sub(r'https?://[^\s]+', '[URL] ', unificated_text)
    unificated_text = re.sub(r'https?://[^\s]+', '[URL]', unificated_text)
    unificated_text = re.sub(r'\d+[\d,.]*\s', '[NUM] ', unificated_text)
    unificated_text = re.sub(r'\s\d+[\d,.]*', ' [NUM]', unificated_text)
    unificated_text = re.sub(r'\d+[\d,.]*$', '[NUM]', unificated_text)
    unificated_text = re.sub(r'\[NUM\]:\[NUM\]', '[NUM]', unificated_text)
    unificated_text = re.sub(r'\[NUM\]:\[NUM\]', '[NUM]', unificated_text)
    unificated_text = re.sub(r'\[NUM\]-\S+\s', '[NUM] ', unificated_text)
    unificated_text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s', '[EMAIL] ', unificated_text)
    unificated_text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', '[EMAIL]', unificated_text)
    unificated_text = re.sub(r'\s+', ' ', unificated_text).strip()
    return unificated_text
    

def preprocessor(text):
    preprocessed_text = clean_abbreviations(text)
    preprocessed_text = clean_punctuation(preprocessed_text)
    preprocessed_text = clean_stopwords(preprocessed_text)
    preprocessed_text = unification(preprocessed_text)
    return preprocessed_text




def Space_Preprocessed_Tokenizer_Lemmatization(text):
    morph = pymorphy2.MorphAnalyzer(lang = 'ru')
    cleaned_text = cleaner(text)
    preprocessed_text = preprocessor(cleaned_text)
    Tokens = preprocessed_text.split(' ')
    for t in range(len(Tokens)):
        Tokens[t] = (morph.parse(Tokens[t])[0]).normal_form
    return Tokens


def NLTK_Word_Learning_Tokenizer_Lemmatization(text):
    morph = pymorphy2.MorphAnalyzer(lang = 'ru')
    cleaned_text = cleaner(text)
    Tokens = nltk.word_tokenize(cleaned_text)
    for t in range(len(Tokens)):
        Tokens[t] = (morph.parse(Tokens[t])[0]).normal_form
    return Tokens
