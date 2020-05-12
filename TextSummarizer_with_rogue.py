from typing import List

import nltk
import re
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from rouge import rouge_n_summary_level

nltk.download("stopwords")
nltk.download('punkt')


class TextSummarizer:
    def __init__(self, title: str, plot: str, human_synopsis: str):
        self.title = title
        self.plot = plot
        self.human_synopsis = human_synopsis
        self.stopwords = StopWordRemoverFactory().create_stop_word_remover()
        self.stemmer = StemmerFactory().create_stemmer()
        
        self.ringkasan = ""

    def __text_to_sentences(self, text: str) -> List[str]:
        regex = re.compile('\.\n\n|\.\n|\. |\.$')
        sentences = regex.split(text)
        return sentences

    def __stem_sentence(self, sentence: str) -> str:
        return self.stemmer.stem(sentence)

    def __stop_word_removal(self, words: List[str]) -> List[str]:
        temp_words = []
        for word in words:
            if word.lower() in self.title.lower():
                temp_words.append(word)
            else:
                temp = self.stopwords.remove(word)
                if temp:
                    temp_words.append(temp)

        return temp_words

    def __preprocess_text(self, text: str) -> tuple:
        temp_sentences = self.__text_to_sentences(text)
        sentences = []
        preprocessed_sentences = []
        for sentence in temp_sentences:
            if len(sentence) < 2:
                continue

            stemmed_sentence = self.__stem_sentence(sentence.lower())
            tokenized_sentence = nltk.tokenize.word_tokenize(stemmed_sentence)
            removed_stop_word_sentence = self.__stop_word_removal(tokenized_sentence)

            if len(removed_stop_word_sentence) < 2:
                continue

            sentences.append(sentence)
            preprocessed_sentences.append(removed_stop_word_sentence)

        return sentences, preprocessed_sentences

    def __sentence_similarity(self, sent1, sent2):
        """
        calculate the similarity between sentence!
        return distance between sentences
        """
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def __build_similarity_matrix(self, sentences):
        """
        make a matrix to plot the similarity between sentences in a file
        return matrix
        """
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = self.__sentence_similarity(sentences[idx1], sentences[idx2])

        return similarity_matrix

    def summarize(self, top_n=5):
        summarize_text = []

        # Step 1 - text preprocessing
        plot_sentences, plot_pre_sentences = self.__preprocess_text(self.plot)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.__build_similarity_matrix(plot_pre_sentences)

        print(sentence_similarity_martix)
        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        plot_scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = []
        for i in range(len(plot_scores)):
            ranked_sentence.append([plot_scores[i], plot_sentences[i], i])

        ranked_sentence.sort(key=lambda x: x[0], reverse=True)
        top_n = min(top_n, len(plot_sentences))
        summary = ranked_sentence[0:top_n]
        summary.sort(key=lambda x: x[2])
        summary = [i[1] for i in summary]
        summarize_text = ""
        for i in range(top_n):
            summarize_text += "".join(summary[i]) + ". "

        # Step 5 - Offcourse, output the summarize texr
        #return summarize_text

        self.ringkasan = summarize_text
        #print(self.ringkasan)

        human_synopsis, human_prepocessed_synopsis = self.__preprocess_text(self.human_synopsis)
        ringkasan, token_ringkasan = self.__preprocess_text(self.ringkasan)

        result = {"summarize_text": summarize_text, "human_synopsis" : human_synopsis, "human_preprocessed_synopsis" : human_prepocessed_synopsis, "ringkasan": ringkasan, "token_ringkasan": token_ringkasan}
        return result
    @staticmethod
    def generate_from_file(title, plotfilepath, synopsisfilepath):
        plot = ""
        synopsis = ""
        with open(plotfilepath, "r") as plot_file:
            plot = plot_file.read()
        with open(synopsisfilepath, "r") as synopsis_file:
            synopsis = synopsis_file.read()

        ts = TextSummarizer(title, plot, synopsis)
        return ts.summarize()


if __name__ == '__main__':
    result = TextSummarizer.generate_from_file('Ada Apa Dengan Cinta', 'dataset/plot/1.txt', 'dataset/synopsis/1.txt')
    print(result.get("summarize_text"))

    stemmed_synopsis = result.get('human_preprocessed_synopsis')
    stemmed_summary = result.get('token_ringkasan')

    #print(stemmed_summary,stemmed_synopsis)
    print('Summary level:')
    _, _, rouge_1 = rouge_n_summary_level(result.get("token_ringkasan"), result.get("human_preprocessed_synopsis"), 1)
    print('ROUGE-1: %f' % rouge_1)


