import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# a)
''''
(1a) Load the SpotifyFeatures.csv file and report the number of samples (songs) as well as the number of
features (song properties) in the dataset. Hint: you may use the Python module Pandas and its function
read_csv.
'''

# Loading/opening data
spotify = pd.read_csv('SpotifyFeatures.csv')

# Finding how many samples (songs) there is
rows = len(spotify.axes[0])
print('Number of songs in Spotify-file: ', rows)

# Finding number of features
cols = len(spotify.axes[1])
print('Number of features in Spotify-file: ', cols)

# b)
'''
(1b) You will be working with samples from two genres namely ’Pop’ and ’Classical’. Retrieve all samples
belonging to the two genres and create labels for the samples i.e: ’Pop’ = 1, ’Classical’ = 0. Report how
many samples belongs to the two classes. Working with all features is not always the best solution since
it increases the computational cost and some of them may be useless for the task. For this dataset you
should be able to separate the two classes by using two features, namely ’liveness’ and ’loudness’.
'''

# Get the unique genres
#unique_genres = spotify['genre'].unique()

# Retrieving the genres 'Pop' and 'Classical'
filtered_spotify = spotify[spotify['genre'].isin(['Pop', 'Classical'])].copy()

# Finding number of sampels in each genre
pop_count = (filtered_spotify['genre'] == 'Pop').sum()
classical_count = (filtered_spotify['genre'] == 'Classical').sum()
print('how many pop-songs there are: ', pop_count)
print('how many country-songs there are', classical_count)

# Renaming Pop and Country from strings to binary numbers
filtered_spotify.loc[:, 'label'] = filtered_spotify['genre'].apply(lambda x: 1 if x == 'Pop' else 0)


# c)
'''
(1c) From the reduced dataset, make 2 numpy arrays. The first array will be the matrix with songs along the
rows and songs’ features ("liveness" and "loudness") as columns. This will be the input of our machine
learning method. The second array will the vector with the songs’ genre (labels or target we want to
learn). Create a training and test set by splitting the dataset. Use an 80% 20% split between the training
and test set. Split the data per class so that you keep the same class distribution in the training and test
set.
'''

# Making two numpy arrays, one for the features' information, and one for the labels (zero's and one's)
X = filtered_spotify[['liveness', 'loudness']].values
y = filtered_spotify['label'].values

# Separating data and finding indexes of samples belonging to Classical (0) and Pop (1)
classical0_index = np.where(y == 0)[0]
pop0_index = np.where(y == 1)[0]

# Shuffling to randomize data to keep it unbiased during splitting of data
np.random.shuffle(classical0_index)
np.random.shuffle(pop0_index)

# Testing 20% of the data, and keeping 80% to train
test_size = 0.2

# Splitting sets, and determining how many samples will go into each set (train and test)
split_classical = int(len(classical0_index) * (1 - test_size))
split_pop = int(len(pop0_index) * (1 - test_size))

# Using 20% of shuffled classical-data for testing and the remaninig 80% shuffeled classical-data for training
train_classical = classical0_index[:split_classical]
test_classical = classical0_index[split_classical:]

# Using 20% of shuffled pop-data for testing and the remaninig 80% shuffeled pop-data for training
train_pop = pop0_index[:split_pop]
test_pop = pop0_index[split_pop:]

# Combining training sets from both classical and pop
train_samples = np.concatenate([train_classical, train_pop])

# Combining test sets from both classical and pop
test_samples = np.concatenate([test_classical, test_pop])

# Mixing samples so they are not grouped in classical and pop
np.random.shuffle(train_samples)
np.random.shuffle(test_samples)

# Creating the final training and test sets
X_train, X_test = X[train_samples], X[test_samples]  # Split the features into training and test sets
y_train, y_test = y[train_samples], y[test_samples]  # Split the labels into training and test sets

# d)
''''
(1d) [Bonus.] Plot the samples on the liveness vs loudness plane,
with a different color for each class. From the
plot, will the classification be an easy task? why?
'''

if __name__ == '__main__':
    # Plotting Classical (0) songs
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Classical (0)', alpha=0.5)
    # Plotting Pop (1) songs
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], facecolors='none', edgecolors='red', label='Pop (1)', alpha=0.5)

    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.title('Liveness vs Loudness')
    plt.legend()
    plt.show()
