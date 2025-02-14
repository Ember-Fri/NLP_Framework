import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def vectorize(words, unique):
    """
    name: vectorize
    parameter: words: list of a users words, unique list of all unique words
    returns: a vector with counter values that give a value to how many of the certain unique word was in the user list
    does: convert into a vectors from user words and unique words
    """
    return [Counter(words)[word] for word in unique]


def mag(v):
    """
    name: mag
    parameters: vector like list
    returns: magnitude of a vector
     """
    return (sum([i ** 2 for i in v])) ** .5


def dot(u, v):
    """
    name: dot
    parameters: u, v both vector like lists of same size
    returns: dot product of two vectors
    """
    return sum([i * j for i, j in zip(u, v)])


def cosine_similarity(u, v):
    """
    name: cosine_similarity
    parameters: u, v both vector like lists of same size
    returns cosine similarity between two vectors
    """
    if mag(u) != 0 and mag(v) != 0:
        return dot(u, v) / (mag(u) * mag(v))
    else:
        return


def cosine_similarity_array(dct, unique):
    """
    name: cosine_similarity_array
    parameters: dct: dictionary, unique, a set of unique words
    returns nothing, plots a heatmap of which keys are most similar to each other
    """
    lst = list(dct.items())
    arr = np.ones((len(lst), len(lst)), dtype=float)
    x_labels = []

    for i in range(len(lst)):
        vi = vectorize(lst[i][1], unique)
        x_labels.append(lst[i][0])
        for j in range(i + 1, len(lst)):
            vj = vectorize(lst[j][1], unique)

            arr[i, j] = cosine_similarity(vi, vj)
            arr[j, i] = arr[i, j]

    sns.heatmap(arr, xticklabels=x_labels, yticklabels=x_labels)
    plt.show()
    return


def unique_words_in_dct(dct, most_common=None):
    """
    name: unique_words_in_dct
    parameter: dictionary a dictionary with values as list of words, most_common optional is in an
    return: returns a unique set of words
    """
    words = []
    if most_common is not None:
        for value in dct.values():
            top_n = Counter(value).most_common(most_common)
            top_n = [item[0] for item in top_n]
            value = [word for word in value if word in top_n]
            words.extend(value)

    else:
        for value in dct.values():
            words.extend(value)

    return set(words)

def unique_words_in_dct(dct, most_common=None):
    """
    name: unique_words_in_dct
    parameter: dictionary a dictionary with values as list of words, most_common optional is in an
    return: returns a unique set of words
    """
    words = []
    if most_common is not None:
        for value in dct.values():
            top_n = Counter(value).most_common(most_common)
            top_n = [item[0] for item in top_n]
            value = [word for word in value if word in top_n]
            words.extend(value)

    else:
        for value in dct.values():
            words.extend(value)

    return set(words)

def cosine_simil(self):
    '''
    Calculate the cosine similarity between the text files
    '''

    # get the word count for each text file
    labels = list(self.data['wordcount'].keys())
    words = set()

    for label in labels:
        words.update(self.data['wordcount'][label].keys())

    words_sorted = sorted(words)

    vectors = []
    for label in labels:
        vectors.append([self.data['wordcount'][label][word] for word in words_sorted])

    similarity_matrix = cosine_similarity(vectors)

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels,
                cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    plt.title('Cosine Similarity Heatmap Based on Word Frequency')
    plt.xlabel('Text Files')
    plt.ylabel('Text Files')
    plt.tight_layout()
    plt.show()
