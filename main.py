from nltk.tokenize import word_tokenize
from lxml import etree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from string import punctuation


def lemmatize_text(_text):
    lemmatized = []
    lemmatizer = WordNetLemmatizer()
    for t in _text:
        lemmatized.append(lemmatizer.lemmatize(t))

    return lemmatized


def remove_stopwords(_text):
    stop = stopwords.words('english')
    pun = list(punctuation)
    return [t for t in _text if t not in stop and t not in pun]


def pos_tag_nouns(_text):
    nouns = []
    for t in _text:
        tagged = pos_tag([t])[0]
        if tagged[1] == "NN":
            nouns.append(t)

    return nouns


def vectorize(_text):
    vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True, analyzer='word', ngram_range=(1, 1))
    vector = vectorizer.fit_transform(_text)
    tf_idf = []
    for i in range(len(vector.toarray())):
        tf_idf.append([])
        for (score, word) in zip(vector.toarray()[i], vectorizer.get_feature_names()):
            tf_idf[i].append((score, word))

    return tf_idf


def process_stories(_root):
    _stories = []
    all_texts = []
    all_headlines = []
    for news in _root[0]:
        all_headlines.append(news[0].text)
        text = word_tokenize(news[1].text.lower())
        text = lemmatize_text(text)
        text = remove_stopwords(text)
        text = pos_tag_nouns(text)
        all_texts.append(" ".join(text))

    tf_idf = vectorize(all_texts)
    for i in range(len(all_headlines)):
        _stories.append((all_headlines[i], tf_idf[i]))

    return _stories


root = etree.parse(input()).getroot()
stories = process_stories(root)

for story in stories:
    story[1].sort(reverse=True)
    print(f"{story[0]}:")
    print(" ".join([x[1] for x in story[1]][:5]))
    print()
