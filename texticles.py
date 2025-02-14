from collections import defaultdict, Counter
import textblob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import csv
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class Textastic:

    def __init__(self):
        """ Constructor """
        self.data = defaultdict(dict)
        self.categories = {"Before Chernobyl": [], "After Chernobyl": []}
        self.stopwords = set()

    def load_stop_words(self, stopfile):
        """
        Load stopwords from a file and store them in self.stopwords.
        :param stopfile: Path to the stopword file.
        """
        try:
            with open(stopfile, 'r', encoding='utf-8') as file:
                self.stopwords = set(file.read().splitlines())
        except Exception as e:
            raise ValueError(f"Error loading stopwords: {e}")

    def preprocess_text(self, text):
        """
        Preprocess text by cleaning and removing stopwords.
        :param text: Raw text.
        :return: Cleaned text.
        """
        # Remove punctuation and lowercase the text
        cleaned = ''.join(char.lower() if char.isalnum() or char.isspace() else ' ' for char in text)

        # Remove stopwords
        if self.stopwords:
            cleaned = ' '.join(word for word in cleaned.split() if word not in self.stopwords)

        return cleaned

    def default_parser(self, filename):
        """
        Default parser for a standard text file.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = self.preprocess_text(text)
        wordcount = Counter(cleaned_text.split())
        numwords = len(cleaned_text.split())
        return {'wordcount': wordcount, 'numwords': numwords}

    def json_parser(filename):
        f = open(filename)
        raw = json.load(f)
        text = raw['text']
        words = text.split(" ")
        wc = Counter(words)
        num = len(words)

        return {'wordcount':wc, 'numwords':num}

    def csv_parser(self, filename, column):
        """
        Custom parser for CSV files.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            text = ' '.join(row[column] for row in reader if column in row)
            cleaned_text = self.preprocess_text(text)
            wordcount = Counter(cleaned_text.split())
            numwords = len(cleaned_text.split())
            return {'wordcount': wordcount, 'numwords': numwords}

    def load_text(self, filename, label=None, parser=None, category=None, **kwargs):
        """
        Load and categorize a text file.
        :param filename: Path to the text file.
        :param label: Label for the text file.
        :param parser: Optional custom parser function.
        :param category: Text category ("Before Chernobyl" or "After Chernobyl").
        :param kwargs: Additional arguments for custom parsers (e.g., column name for CSV).
        """
        # Use the specified parser, or default if none is provided
        if parser is None:
            results = self.default_parser(filename)
        else:
            results = parser(filename, **kwargs)
    
        # Assign a default label if none is provided
        if label is None:
            label = filename
    
        # Store results in the data structure
        for k, v in results.items():
            self.data[k][label] = v
    
        # Categorize the text if a category is specified
        if category in self.categories:
            self.categories[category].append(label)
        else:
            raise ValueError(f"Invalid category: {category}")

    def sentiment_analysis(self):
        """
        Perform sentiment analysis on the text data.
        """
        # Perform sentiment analysis on each text
        print(self.data['wordcount'].keys())
        for label, wordcount in self.data['wordcount'].items():
            text = ' '.join([word for word in wordcount for _ in range(wordcount[word])])
            blob = textblob.TextBlob(text)
            self.data['polarity'][label] = blob.sentiment.polarity
            self.data['subjectivity'][label] = blob.sentiment.subjectivity

    def wordcount_sankey(self, word_list=None, k=5):
        """
        Create a Sankey diagram mapping text files to words.
        :param word_list: Optional list of specific words to include.
        :param k: Number of top words to use if word_list is not provided.
        """
        sources = []
        targets = []
        values = []

        for label, wordcount in self.data['wordcount'].items():
            if not word_list:
                # Get the top-k words
                top_words = wordcount.most_common(k)
            else:
                # Filter only the specified words
                top_words = [(word, wordcount[word]) for word in word_list if word in wordcount]

            for word, count in top_words:
                sources.append(label)
                targets.append(word)
                values.append(count)

        # Build Sankey diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(sources + targets)),  # Unique labels for nodes
            ),
            link=dict(
                source=[list(set(sources + targets)).index(src) for src in sources],
                target=[list(set(sources + targets)).index(tgt) for tgt in targets],
                value=values,
            )
        ))

        fig.update_layout(title_text="Nuclear Analysis Word Count Sankey Diagram", font_size=10)
        fig.show()

    def subjectivity_polarity_viz(self):
        """
        plot sentiment analysis data vs polarity and subjectivity per file
        """
        self.sentiment_analysis()
        num_plots = len(self.data['polarity'])
        rows = (num_plots + 1) // 2  # Calculate the number of rows needed
        fig, axs = plt.subplots(rows, 2, figsize=(15, 5 * rows))

        for i, (label, polarity) in enumerate(self.data['polarity'].items()):
            subjectivity = self.data['subjectivity'][label]

            row = i // 2
            col = i % 2
            axs[row, col].bar(['Polarity', 'Subjectivity'], [polarity, subjectivity])
            axs[row, col].set_title(f"{label} - Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")
            axs[row, col].set_ylim(-.05, .75)  # Set fixed y-axis limits

        # Hide any unused subplots
        for j in range(i + 1, rows * 2):
            fig.delaxes(axs[j // 2, j % 2])

        plt.tight_layout()
        plt.show()

    def cos_similarity_plt(self):
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
                    cmap='Reds', annot=True, fmt=".2f", cbar=True)
        plt.title('Cosine Similarity Heatmap Based on Word Frequency')
        plt.xlabel('Text Files')
        plt.ylabel('Text Files')
        plt.show()


