import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pickle
import nltk
nltk.download('wordnet')
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')

class Sensationalism():
  #def lemmatize_stemming(self, text):
    #stemmer = SnowballStemmer('english')
    #return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

  def preprocess(self, raw_news):
    import nltk
    
    # 1. Remove non-letters/Special Characters and Punctuations
    news = re.sub("[^a-zA-Z]", " ", str(raw_news))
    
    # 2. Convert to lower case.
    news =  news.lower()
    
    # 3. Tokenize.
    news_words = nltk.word_tokenize( news)
    
    # 4. Convert the stopwords list to "set" data type.
    stops = set(nltk.corpus.stopwords.words("english"))
    
    # 5. Remove stop words. 
    words = [w for w in  news_words  if not w in stops]
    
    # 6. Lemmentize 
    wordnet_lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
    
    # 7. Stemming
    stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem ]
    
    # 8. Join the stemmed words back into one string separated by space, and return the result.
    return " ".join(stems)

  
  def predict(text):
    model = pickle.load(open('/content/drive/My Drive/AlternusVeraDataSets2019/FinalExam/Transformers/models/neladatamodel.pkl','rb'))
    sensa = Sensationalism()
    sensa.preprocess(text)
    prediction = model.predict([text])
    print(prediction)
    prob = model.predict_proba([text])[:,1]
    #print(prob)
    return bool(prediction)
