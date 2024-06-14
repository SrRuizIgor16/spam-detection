import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessing:
    """
    A class used to preprocess data for spam detection and to generate various plots for data visualization.

    Attributes
    ----------
    data_path : str
        Path to the raw data file.
    plot_path : str
        Directory where the plots will be saved.
    data : DataFrame
        The loaded raw data.
    raw_data : DataFrame
        A copy of the raw data.
    """

    DATA_PATH = '../data/raw/spam.csv'
    PLOT_PATH = '../data/plots/'
    PROCESSED_DATA_PATH = '../data/processed/data.csv'
    COLUMNS_TO_DROP = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    COLUMN_NAMES = ['IsSpam', 'Message']

    def __init__(self, data_path=DATA_PATH, plot_path=PLOT_PATH, encoding='latin1'):
        """
        Initializes the DataPreprocessing class by loading the data and downloading necessary NLTK resources.

        Parameters
        ----------
        data_path : str
            Path to the raw data file.
        plot_path : str
            Directory where the plots will be saved.
        encoding : str
            Encoding format for reading the CSV file.
        """
        self.data_path = data_path
        self.plot_path = plot_path
        self.data = self.load_data(encoding)
        self.raw_data = self.data.copy() if self.data is not None else None
        nltk.download('punkt')
        nltk.download('stopwords')

    def __str__(self):
        return 'DataPreprocessing class'

    def __repr__(self):
        return 'DataPreprocessing class'

    def load_data(self, encoding):
        """
        Loads data from a CSV file.

        Parameters
        ----------
        encoding : str
            Encoding format for reading the CSV file.

        Returns
        -------
        DataFrame
            The loaded data.
        """
        try:
            data = pd.read_csv(self.data_path, encoding=encoding)
            print('Data loaded successfully')
            return data
        except Exception as e:
            print('Error loading data:', e)
            return None

    def preprocess_data(self):
        """
        Preprocesses the data by dropping unnecessary columns, renaming columns, filling missing values, and mapping labels.
        """
        if self.data is not None:
            self.drop_columns(self.COLUMNS_TO_DROP)
            self.rename_columns(self.COLUMN_NAMES)
            self.fill_na('IsSpam', 0)
            self.drop_na()
            self.data['IsSpam'] = self.data['IsSpam'].map({'ham': 0, 'spam': 1})
        else:
            print('Data not loaded. Cannot preprocess data.')

    def drop_columns(self, columns):
        """
        Drops specified columns from the data.

        Parameters
        ----------
        columns : list
            List of column names to drop.
        """
        self.data.drop(columns=columns, axis=1, inplace=True)

    def rename_columns(self, columns):
        """
        Renames columns of the data.

        Parameters
        ----------
        columns : list
            List of new column names.
        """
        self.data.columns = columns

    def fill_na(self, column, value):
        """
        Fills missing values in a specified column with a given value.

        Parameters
        ----------
        column : str
            The name of the column.
        value : any
            The value to fill missing entries with.
        """
        self.data[column].fillna(value, inplace=True)

    def drop_na(self):
        """
        Drops all rows with any missing values.
        """
        self.data.dropna(inplace=True)

    def save_data(self, path=PROCESSED_DATA_PATH):
        """
        Saves the preprocessed data to a CSV file.

        Parameters
        ----------
        path : str
            The path where the processed data will be saved.
        """
        if self.data is not None:
            try:
                self.data.to_csv(path, index=False)
                print('Data saved successfully')
            except Exception as e:
                print('Error saving data:', e)
        else:
            print('Data not loaded. Cannot save data.')

    def save_plot(self, title):
        """
        Saves the current plot with a given title.

        Parameters
        ----------
        title : str
            The title of the plot (used as the filename).
        """
        try:
            plt.savefig(self.plot_path + title + '.png')
            print(f'Plot "{title}" saved successfully')
        except Exception as e:
            print(f'Error saving plot "{title}":', e)
        plt.close()

    def count_messages(self, data):
        """
        Generates and saves a bar plot showing the distribution of the target variable.

        Parameters
        ----------
        data : DataFrame
            The data to plot.
        """
        if data is not None:
            plt.figure(figsize=(8, 6))
            data['IsSpam'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
            plt.title('Distribution of the target variable')
            plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'], rotation=0)
            self.save_plot('count_messages')
        else:
            print('Data not loaded. Cannot plot count messages.')

    def clean_text(self, data, isSpam):
        """
        Cleans the text data by tokenizing, converting to lowercase, and removing stopwords and non-alphabetic words.

        Parameters
        ----------
        data : DataFrame
            The data containing messages.
        isSpam : int
            The label indicating whether to clean spam (1) or ham (0) messages.

        Returns
        -------
        list
            A list of cleaned words.
        """
        if data is not None:
            messages = data[data['IsSpam'] == isSpam]['Message']
            messages = messages.str.lower()
            words = word_tokenize(' '.join(messages))
            words = [word for word in words if word.isalpha()]
            return [word for word in words if word not in stopwords.words('english')]
        else:
            print('Data not loaded. Cannot clean text.')
            return []

    def distribution_plot(self, data, isSpam):
        """
        Generates and saves a histogram showing the distribution of message lengths.

        Parameters
        ----------
        data : DataFrame
            The data to plot.
        isSpam : int
            The label indicating whether to plot spam (1) or ham (0) messages.
        """
        if data is not None:
            messages = data[data['IsSpam'] == isSpam]['Message'].str.len()
            plt.hist(messages, bins=20, alpha=0.7, label=('spam' if isSpam else 'ham'))
            plt.legend(loc='upper right')
            plt.title('Distribution of message length for ' + ('spam' if isSpam else 'ham'))
            self.save_plot('distribution_plot_' + ('spam' if isSpam else 'ham'))
        else:
            print('Data not loaded. Cannot plot distribution.')

    def word_cloud(self, data, isSpam):
        """
        Generates and saves a word cloud for the text data.

        Parameters
        ----------
        data : DataFrame
            The data containing messages.
        isSpam : int
            The label indicating whether to generate a word cloud for spam (1) or ham (0) messages.
        """
        if data is not None:
            words = self.clean_text(data, isSpam)
            if words:
                wordcloud = WordCloud(
                    width=800, height=800,
                    background_color='white',
                    stopwords=set(stopwords.words('english')),
                    min_font_size=10
                ).generate(' '.join(words))
                plt.figure(figsize=(8, 8), facecolor=None)
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.title('Word cloud for ' + ('spam' if isSpam else 'ham'))
                self.save_plot('word_cloud_' + ('spam' if isSpam else 'ham'))
            else:
                print(f"No words to plot for {'spam' if isSpam else 'ham'}.")
        else:
            print('Data not loaded. Cannot plot word cloud.')

    def word_frequency(self, data, isSpam):
        """
        Generates and saves a bar plot showing the word frequency.

        Parameters
        ----------
        data : DataFrame
            The data to plot.
        isSpam : int
            The label indicating whether to plot word frequency for spam (1) or ham (0) messages.
        """
        if data is not None:
            words = self.clean_text(data, isSpam)
            if words:
                words_freq = pd.Series(words).value_counts()

                # Create a plot for the most common words in spam messages
                plt.figure(figsize=(12, 6))
                words_freq[:20].plot(kind='bar', color=('salmon' if isSpam else 'lightblue'))
                plt.title('Word frequency for ' + ('spam' if isSpam else 'ham'))
                plt.xlabel('Words')
                plt.ylabel('Frequency')
                self.save_plot('word_frequency_' + ('spam' if isSpam else 'ham'))
            else:
                print(f"No words to plot for {'spam' if isSpam else 'ham'}.")
        else:
            print('Data not loaded. Cannot plot word frequency.')

    def plots(self, data):
        """
        Generates and saves all relevant plots for the data.

        Parameters
        ----------
        data : DataFrame
            The data to plot.
        """
        self.count_messages(data)
        self.distribution_plot(data, 0)
        self.distribution_plot(data, 1)
        self.word_cloud(data, 0)
        self.word_cloud(data, 1)
        self.word_frequency(data, 0)
        self.word_frequency(data, 1)

def main():
    """
    Main function to execute the data preprocessing and plotting.
    """
    data_preprocessing = DataPreprocessing()
    if data_preprocessing.data is not None:
        data_preprocessing.preprocess_data()
        data_preprocessing.save_data()
        data_preprocessing.plots(data_preprocessing.data)

if __name__ == '__main__':
    main()
