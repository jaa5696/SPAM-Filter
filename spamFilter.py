
############################################################
# Imports
############################################################

import collections
import email
import math
import os

############################################################
# SPAM Filter Code
############################################################


def load_tokens(email_path):

    with open(email_path, 'r', encoding="utf-8") as email_contents:
        message = email.message_from_file(email_contents)
        itera = email.iterators.body_line_iterator(message)
        lines = list(itera)

        tokens = [word for line in lines for word in line.split()]

        return tokens


def log_probs(email_paths, smoothing):

    # get all tokens from all path and put it in one list

    all_tokens = [x for path in email_paths for x in load_tokens(path)]

    total = len(all_tokens)

    word_counts = collections.Counter(all_tokens)

    # for word in all_tokens:          # can be more efficient #
    #    word_counts[word] += 1

    total_words = len(word_counts.keys())

    denom = total + (smoothing * (total_words + 1))

    word_prob_dict = {}

    word_prob_dict["<UNK>"] = probability(0, smoothing, denom)

    for key in word_counts:

        count = word_counts[key]

        word_prob_dict[key] = probability(count, smoothing, denom)

    return word_prob_dict


def probability(count, alpha, denominator):

    if count == 0:  # <UNK> probability
        prob = alpha / denominator

        return math.log(prob)

    else:  # probability for words in training

        prob = (count + alpha) / denominator

        return math.log(prob)


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):

        spam_dire = os.path.join(os.getcwd(), spam_dir)

        ham_dire = os.path.join(os.getcwd(), ham_dir)

        spam_files = os.listdir(spam_dire)

        ham_files = os.listdir(ham_dire)

        spam_paths = [os.path.join(spam_dire, x) for x in spam_files]

        ham_paths = [os.path.join(ham_dire, x) for x in ham_files]

        self.spam_logs = log_probs(spam_paths, smoothing)

        self.ham_logs = log_probs(ham_paths, smoothing)

        self.prob_spam = math.log(len(spam_files)/(len(spam_files) + len(ham_files)))

        self.prob_not_spam = math.log(len(ham_files)/(len(spam_files) + len(ham_files)))
    
    def is_spam(self, email_path):

        sum_spam_logs = []

        sum_ham_logs = []

        file_tokens = load_tokens(email_path)

        file_word_counts = collections.Counter()

        for token in file_tokens:

            file_word_counts[token] += 1

        for word in file_word_counts:

            if word in self.spam_logs:

                sum_spam_logs.append(file_word_counts[word] * self.spam_logs[word])

            else:

                sum_spam_logs.append(file_word_counts[word] * self.spam_logs["<UNK>"])

            if word in self.ham_logs:

                sum_ham_logs.append(file_word_counts[word] * self.ham_logs[word])

            else:

                sum_ham_logs.append(file_word_counts[word] * self.ham_logs["<UNK>"])

        total_prob_spam = sum(sum_spam_logs) + self.prob_spam

        total_prob_ham = sum(sum_ham_logs) + self.prob_not_spam

        if total_prob_spam > total_prob_ham:

            return True

        else:

            return False

    def most_indicative_spam(self, n):

        most_ind = []

        indicative_dict = {}
        for word in self.spam_logs:

            if (word != "<UNK>") and (word in self.ham_logs):

                prob_w = math.log(pow(math.e, self.spam_logs[word]) + pow(math.e, self.ham_logs[word]))

                indicative_dict[word] = self.spam_logs[word] - prob_w

        ind = indicative_dict.items()

        sorted_ind = sorted(ind, key=lambda words: words[1], reverse=True)

        for i in range(n):

            most_ind.append(sorted_ind[i][0])

        return most_ind

    def most_indicative_ham(self, n):
        most_ind = []

        indicative_dict = {}
        for word in self.spam_logs:

            if (word != "<UNK>") and (word in self.ham_logs):
                prob_w = math.log(pow(math.e, self.spam_logs[word]) + pow(math.e, self.ham_logs[word]))

                indicative_dict[word] = self.ham_logs[word] - prob_w

        ind = indicative_dict.items()

        sorted_ind = sorted(ind, key=lambda words: words[1], reverse=True)

        for i in range(n):
            most_ind.append(sorted_ind[i][0])

        return most_ind