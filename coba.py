# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

swfactory = StopWordRemoverFactory()
stopword = swfactory.create_stop_word_remover()

output = stopword.remove('aku adalah saya ketika dia menjadi kamu')
print(swfactory.get_stop_words())
print(output)

# stemming process
sentence = 'Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan'
output   = stemmer.stem(sentence)

print(output)
# ekonomi indonesia sedang dalam tumbuh yang bangga

print(stopword.remove(stemmer.stem('Mereka meniru-nirukannya')))
# mereka tiru