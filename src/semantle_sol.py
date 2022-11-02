import pickle
import numpy as np
import sys
import os

def load_object(filepath):
    fileObj = open(f'{filepath}', 'rb')
    obj = pickle.load(fileObj)
    fileObj.close()
    return obj

def cosine_distance(list_1, list_2):
  cos_sim = np.dot(list_1, list_2) / (np.linalg.norm(list_1) * np.linalg.norm(list_2))
  return cos_sim

def all_cosines(guess, words):
    # num_words x features @ features x 1 = num_words x 1
    numerator = words @ guess
    # words_matrix @ words_matrix.T --> the squares of the matrix are on the diagonal
    # dim: num_words x features \w features x num_words --> num_words x 1
    norms = np.sqrt(np.einsum('ij,ji->i', words, words.T))
    # elem-multiply by norm of guess, denominator --> (num_words x 1)
    denominator = norms * np.linalg.norm(guess, 2)
    cosines = numerator / denominator
    return np.round(cosines, 3)

def most_similar(n, matrix, idx2word):
    sorted = np.argsort(matrix, axis=-1)
    return [(idx2word[i], matrix[i]) for i in sorted[len(sorted)-n:len(sorted)]]

def do_run():
    guesses = []
    word2idx = load_object('save/worddict.dic')
    idx2word = load_object('save/indices.list')
    matrix = load_object('save/witcher.mat')
    hidden_word = np.random.choice(idx2word)
    hidden_vec = matrix[word2idx[np.random.choice(idx2word)]]

    n_hints = 0
    top = most_similar(25, all_cosines(hidden_vec, matrix), idx2word)
    while True:
        data = input("Guess: ")
        os.system('clear')
        if data == "!quit":
            sys.exit()
        elif data == "!resign":
            print(f"The secret word was {hidden_word}")
        elif data == "!hint" :
            # TODO: What to do when the user wants a hint?
            hint = top[n_hints]
            guesses.append(hint)
            n_hints += 1
        elif data not in idx2word:
            sys.stdout.write("Out of Vocabulary word \n")
        elif data == hidden_word:
            print(f"You won! The hidden word was {hidden_word}")
            sys.exit()
        else:
            cos = cosine_distance(matrix[word2idx[data]], hidden_vec)
            if (data, cos) not in guesses:
                guesses.append((data, cos))
        for idx, (word, cos) in enumerate(sorted(guesses, key=lambda tup: tup[1])):
            if word == data:
                print(f"{word} {np.round(cos, 3)}**")
            else:
                print(f"{word} {np.round(cos, 3)}")


do_run()
