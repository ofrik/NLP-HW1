"""
A spell checker for HW 1 in NLP course in BGU.
The spell checker create an error model from a given error file.
The spell checker create a ngram language model form given files, normalizing the text by splitting it to sentences
using pretty simple regular expression, then replacing the newlines in white space, lower all the capital letters and
removing all the none letter and white spaces from the sentence.
I used laplace smoothing to handle out of vocabulary words.
"""
import re
import itertools
import sys
import time
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %d ms' % (f.func_name, (time2 - time1) * 1000.0)
        return ret

    return wrap


def learn_language_model(files, n=3, lm=None):
    """ Returns a nested dictionary of the language model based on the
    specified files. For Text Normalization i used first, lower the capital letters.
    second, i removed all the non-letter characters. i choose to do these normalizations to reduce the size of
    the vocabulary as much as possible. it will still be a valid language model that will catch most of the
    non-word and real-word. i am also processing the text data as sentences using regular expression

    Example of the returned dictionary for the text 'w1 w2 w3 w1 w4' with a
    tri-gram model:
    tri-grams:
    <> <> w1
    <> w1 w2
    w1 w2 w3
    w2 w3 w1
    w3 w1 w4
    w1 w4 <>
    w4 <> <>
    and returned language model is:
    {
    w1: {'':1, 'w2 w3':1},
    w2: {w1:1},
    w3: {'w1 w2':1},
    w4:{'w3 w1':1},
    '': {'w1 w4':1, 'w4':1}
    }

    Args:
          files (list): a list of files (full path) to process.
          n (int): length of ngram, default 3.
          lm (dict): update and return this dictionary if not None.
                     (default None).

    Returns:
        dict: a nested dict {str:{str:int}} of ngrams and their counts.
    """

    if lm is not None:
        ngrams = lm
    else:
        ngrams = {}

    for file in files:
        with open(file, "r") as f:
            # for line in _sentence_tokenizer(f.read()):
            for line in f.readlines():
                prev_words = []
                if line == "":
                    continue
                cleaned_line = _normalize_text(line)
                words = [x for x in cleaned_line.split(" ") if x != ""]
                words = words + [""] * (n - 1)
                for word in words:
                    if word not in ngrams:
                        ngrams[word] = {}
                    if " ".join(prev_words) not in ngrams[word]:
                        ngrams[word][" ".join(prev_words)] = 0
                    ngrams[word][" ".join(prev_words)] = ngrams[word][" ".join(prev_words)] + 1
                    prev_words.append(word)
                    if len(prev_words) == n:
                        prev_words.pop(0)
    print("Created language model %s-gram" % n)
    return ngrams


def _sentence_tokenizer(s):
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', s)


def _normalize_text(s):
    """
    clean the string without capital letters from any non letter or white space characters
    :param str: any string
    :return: a string with only letters and white spaces
    """
    s = re.sub(r"\n", " ", s)
    return re.sub(r"[^a-z ]+", '', s.lower())


def create_error_distribution(errors_file, lexicon):
    """ Returns a dictionary {str:dict} where str is in:
    <'deletion', 'insertion', 'transposition', 'substitution'> and the inner dict {tupple: float} represents the confution matrix of the specific errors
    where tupple is (err, corr) and the float is the probability of such an error. Examples of such tupples are ('t', 's'), ('-', 't') and ('ac','ca').
    Notes:
        1. The error distributions could be represented in more efficient ways.
           We ask you to keep it simpel and straight forward for clarity.
        2. Ultimately, one can use only 'deletion' and 'insertion' and have
            'sunstiturion' and 'transposition' derived. Again,  we use all
            four explicitly in order to keep things simple.
    Args:
        errors_file (str): full path to the errors file. File format mathces
                            Wikipedia errors list.
        lexicon (dict): A dictionary of words and their counts derived from
                        the same corpus used to learn the language model.

    Returns:
        A dictionary of error distributions by error type (dict).

    """

    error_dist = {"deletion": {}, "insertion": {}, "transposition": {}, "substitution": {}}
    with open(errors_file, "r") as f:
        text = f.read()
        for line in text.split("\n"):
            line = line.lower()
            if line == "":
                continue
            error_word, correct_words = line.split("->")
            error_word = _normalize_text(error_word).strip()
            correct_words = correct_words.split(", ")
            for correct_word in correct_words:
                correct_word = _normalize_text(correct_word).strip()
                j = 0
                i = 0
                while j < len(correct_word) - 1 and i < len(error_word) - 1:
                    xy_correct = correct_word[j:j + 2]
                    x_correct = correct_word[j:j + 1]
                    xy_error = error_word[i:i + 2]
                    x_error = error_word[i:i + 1]
                    letters_left_correct = len(correct_word) - 1 - j
                    letters_left_error = len(error_word) - 1 - i
                    if xy_correct == xy_error[::-1] and xy_correct != xy_error:
                        # transposition
                        tpl = (xy_error, xy_correct)
                        if tpl not in error_dist["transposition"]:
                            error_dist["transposition"][tpl] = 0
                        error_dist["transposition"][tpl] = error_dist["transposition"][tpl] + 1
                        j += 2
                        i += 2
                        continue
                    if x_correct != x_error and i == j and i == 0 and letters_left_correct != letters_left_error:
                        # substitution
                        tpl = ('', x_correct)
                        if tpl not in error_dist["substitution"]:
                            error_dist["substitution"][tpl] = 0
                        error_dist["substitution"][tpl] = error_dist["substitution"][tpl] + 1
                        i += 1
                        continue
                    if x_correct != x_error and letters_left_correct == letters_left_error:
                        # substitution
                        tpl = (x_error, x_correct)
                        if tpl not in error_dist["substitution"]:
                            error_dist["substitution"][tpl] = 0
                        error_dist["substitution"][tpl] = error_dist["substitution"][tpl] + 1
                        j += 1
                        i += 1
                        continue
                    if x_correct == x_error and xy_correct != xy_error and letters_left_error < letters_left_correct:
                        # deletion
                        tpl = (x_error, xy_correct)
                        if tpl not in error_dist["deletion"]:
                            error_dist["deletion"][tpl] = 0
                        error_dist["deletion"][tpl] = error_dist["deletion"][tpl] + 1
                        i += 1
                        j += 2
                        continue
                    if x_correct == x_error and xy_correct != xy_error and letters_left_error > letters_left_correct:
                        # insertion
                        tpl = (xy_error, x_correct)
                        if tpl not in error_dist["insertion"]:
                            error_dist["insertion"][tpl] = 0
                        error_dist["insertion"][tpl] = error_dist["insertion"][tpl] + 1
                        i += 2
                        j += 1
                        continue
                    j += 1
                    i += 1

    def _get_count(s):
        return sum([word.count(s) * d[""] for word, d in lexicon.items()])

    chars = list("abcdefghijklmnopqrstuvwxyz")
    subs = list(itertools.combinations(chars, 2))
    trans = [("".join(x), "".join(x[::-1])) for x in subs]
    trans.extend([("".join(x[::-1]), "".join(x)) for x in subs])
    dels = [(x[0], "".join(x)) for x in itertools.product(*[chars + [""], chars])]
    adds = [("".join(x), x[0]) for x in itertools.product(*[chars, chars])]

    for tuple in subs:
        if tuple not in error_dist["substitution"]:
            error_dist["substitution"][tuple] = 0
    for tuple in trans:
        if tuple not in error_dist["transposition"]:
            error_dist["transposition"][tuple] = 0
    for tuple in dels:
        if tuple not in error_dist["deletion"]:
            error_dist["deletion"][tuple] = 0
    for tuple in adds:
        if tuple not in error_dist["insertion"]:
            error_dist["insertion"][tuple] = 0

    total_errors = sum(sum(error_dist[key].values()) for key in error_dist)

    for (err, corr), value in error_dist["deletion"].items():
        total = _get_count(corr)
        error_dist["deletion"][(err, corr)] = float(value + 1) / (total + total_errors)
    for (err, corr), value in error_dist["insertion"].items():
        total = _get_count(corr)
        error_dist["insertion"][(err, corr)] = float(value + 1) / (total + total_errors)
    for (err, corr), value in error_dist["transposition"].items():
        total = _get_count(corr)
        error_dist["transposition"][(err, corr)] = float(value + 1) / (total + total_errors)
    for (err, corr), value in error_dist["substitution"].items():
        total = _get_count(corr)
        error_dist["substitution"][(err, corr)] = float(value + 1) / (total + total_errors)
    print("Created error model")
    return error_dist


def generate_text(lm, m=15, w=None):
    """ Returns a text of the specified length, generated according to the
     specified language model using the specified word (if given) as an anchor.

     Args:
        lm (dict): language model used to generate the text.
        m (int): length (num of words) of the text to generate (default 15).
        w (str): a word to start the text with (default None)

    Returns:
        A sequrnce of generated tokens, separated by white spaces (str)
    """

    n = max([len(x.split(" ")) for x in lm[lm.keys()[0]].keys()])

    def _choose_given_context(context):
        """
        try to choose a word from the lm given a context, get all the words that has the given context and the number
        instances of that context for the word. choose by the probability compared to other words
        :param context: context for the choosing
        :return: word from the lm that has this context or empty string if there's none
        """
        lst = [(key, d.get(context)) for key, d in lm.items() if context in d]
        total = sum([x[1] for x in lst])
        probas = [float(x[1]) / total for x in lst]
        try:
            return np.random.choice([x[0] for x in lst], 1, probas)[0]
        except:
            return ""

    if w is None:
        w = _choose_given_context("")

    sentence = [w]

    for i in range(1, m):
        context = " ".join(sentence[max(0, i - n):i])
        chosen_word = _choose_given_context(context)
        while chosen_word == "":  # if do not know how to continue the sentence, start a new one
            chosen_word = _choose_given_context("")
        sentence.append(chosen_word)
    return " ".join(sentence)


candidates_cache_2 = {}
candidates_cache_1 = {}
last_error_dist = {}


# @timing
def _generate_candidates_with_proba(w, errors_dist, distance=1):
    """
    create a dictionary of all the candidates and the channel probability for that candidate
    :param w: a word we want to correct
    :param errors_dist: a dictionary of {str:dict} representing the error
                            distribution of each error type (as returned by
                            create_error_distribution()
    :return: dictionary {str:prob} of all the candidates and their probabilities
    """
    if distance == 1:
        if w in candidates_cache_1:
            return candidates_cache_1[w]
    elif distance == 2:
        if w in candidates_cache_2:
            return candidates_cache_2[w]

    correction_proba = {}
    deletions = errors_dist["deletion"].items()
    insertions = errors_dist["insertion"].items()
    substitutions = errors_dist["substitution"].items()
    transpositions = errors_dist["transposition"].items()
    for (err, corr), value in deletions:
        for i in range(len(w)):
            if w[i:i + 1] == err:
                candidate = w[:i] + corr + w[i + 1:]
                correction_proba[candidate] = value
    for (err, corr), value in insertions:
        for i in range(len(w) - 1):
            if w[i:i + 2] == err:
                candidate = w[:i] + corr + w[i + 2:]
                correction_proba[candidate] = value
    for (err, corr), value in substitutions:
        for i in range(len(w)):
            if w[i:i + 1] == err:
                candidate = w[:i] + corr + w[i + 1:]
                correction_proba[candidate] = value
            if i == 0 and err == "":
                candidate = corr + w
                correction_proba[candidate] = value
    for (err, corr), value in transpositions:
        for i in range(len(w) - 1):
            if w[i:i + 2] == err:
                candidate = w[:i] + corr + w[i + 2:]
                correction_proba[candidate] = value
    if distance > 1:
        for i in range(1, distance):
            items = correction_proba.items()
            for word, proba in items:
                tmp_dict = _generate_candidates_with_proba(word, errors_dist)
                for key in tmp_dict:
                    tmp_dict[key] = tmp_dict[key] * proba
                    if key not in correction_proba:
                        correction_proba[key] = tmp_dict[key]

    if distance == 1:
        if w not in candidates_cache_1:
            candidates_cache_1[w] = correction_proba
    elif distance == 2:
        if w not in candidates_cache_2:
            candidates_cache_2[w] = correction_proba
    return correction_proba


def _filter_word_candidates(words_proba, lexicon=None):
    if lexicon is None:
        return words_proba
    keys_to_delete = set(words_proba.keys()) - set(lexicon.keys())
    for key in keys_to_delete:
        words_proba.pop(key, None)
    return words_proba


def correct_word(w, word_counts, errors_dist):
    """ Returns the most probable correction for the specified word, given the specified prior error distribution.

    Args:
        w (str): a word to correct
        word_counts (dict): a dictionary of {str:count} containing the
                            counts  of uniqie words (from previously loaded
                             corpora).
        errors_dist (dict): a dictionary of {str:dict} representing the error
                            distribution of each error type (as returned by
                            create_error_distribution() ).

    Returns:
        The most probable correction (str).
    """
    correction_proba = _filter_word_candidates(_generate_candidates_with_proba(w, errors_dist, 2), word_counts)
    N, V = sum([x[""] for x in word_counts.values()]), len(word_counts)

    best_correction = w
    best_correction_score = 0
    for word, proba in correction_proba.items():
        word_proba = float(word_counts.get(word, {"": 0})[""] + 1) / (N + V)
        score = word_proba * proba
        if score > best_correction_score:
            best_correction_score = score
            best_correction = word
    return best_correction


@timing
def correct_sentence(s, lm, err_dist, c=2, alpha=0.95):
    """ Returns the most probable sentence given the specified sentence, language
    model, error distributions, maximal number of suumed erroneous tokens and likelihood for non-error.

    Args:
        s (str): the sentence to correct.
        lm (dict): the language model to correct the sentence accordingly.
        err_dist (dict): error distributions according to error types
                        (as returned by create_error_distribution() ).
        c (int): the maximal number of tokens to change in the specified sentence.
                 (default: 2)
        alpha (float): the likelihood of a lexical entry to be the a correct word.
                        (default: 0.95)

    Returns:
        The most probable sentence (str)

    """
    sentence_words = [x for x in _normalize_text(s).split(" ") if x != ""]
    sentence_word_candidates = [
        _filter_word_candidates(_generate_candidates_with_proba(w, err_dist, 2), lm).items() for w in
        sentence_words]
    for i in range(len(sentence_words)):
        sentence_word_candidates[i].append((sentence_words[i], alpha))

    def _p(x, w):
        """
        calculate the probability of a given word w is actually supposed to be x
        :param x: the word from a candidate
        :param w: the matching word from the given sentence
        :return: the probability of that w supposed to be x
        """
        if x == w:
            return alpha
        all_cands_for_w = _generate_candidates_with_proba(w, err_dist)
        return (1 - alpha) / len(all_cands_for_w)

    def _generate_sentence_candicate(n):
        """
        generate candidate with up to n errors in the sentence
        :param n: number of maximum errors in a sentence
        :return: list of candidates as array of words
        """
        if n == 0:
            return sentence_words
        indexes_to_check = itertools.combinations(range(len(sentence_words)), n)
        candidates = []
        for indexes in indexes_to_check:
            candidate_words = []
            for i, word in enumerate(sentence_words):
                if i in indexes:
                    candidate_words.append(sentence_word_candidates[i])
                else:
                    candidate_words.append([(word, alpha)])
            candidates += itertools.product(*candidate_words)
        return candidates

    sentence_candidates = _generate_sentence_candicate(c)

    print("number of candidates: %s" % len(sentence_candidates))

    best_candidate = []
    best_candidate_proba = 0

    n = max([len(x.split(" ")) for x in lm[lm.keys()[0]].keys()])

    for j, candidate in enumerate(set(sentence_candidates)):
        candidate_proba = 1
        for i, (word, proba) in enumerate(candidate):
            word_in_org_sentence = sentence_words[i]
            candidate_proba *= _p(word, word_in_org_sentence)
            # candidate_proba *= (
            #     alpha if word == word_in_org_sentence else float(1 - alpha) / len(sentence_word_candidates[i]))
            # candidate_proba *= proba
        if candidate_proba != 0:
            candidate_proba *= evaluate_text(" ".join([x[0] for x in candidate]), n, lm)
        if best_candidate_proba < candidate_proba:
            best_candidate_proba = candidate_proba
            best_candidate = " ".join([x[0] for x in candidate])
    return best_candidate


lm_cache = {}
context_cache = {}


# @timing
def evaluate_text(s, n, lm):
    """ Returns the likelihood of the specified sentence to be generated by the
    the specified language model.

    Args:
        s (str): the sentence to evaluate.
        n (int): the length of the n-grams to consider in the language model.
        lm (dict): the language model to evaluate the sentence by.

    Returns:
        The likelihood of the sentence according to the language model (float).
    """

    def _get_vocab():
        hashkey = str(len(lm)) + "_" + str(n)
        if hashkey not in lm_cache:
            lm_cache[hashkey] = sum([sum(v.values()) for v in lm.values()])
        return lm_cache[hashkey]

    V = _get_vocab()

    s = _normalize_text(s)
    # n = max([len(x.split(" ")) for x in lm[lm.keys()[0]].keys()])
    s_words = [x for x in s.split(" ") if x != ""]

    def _context_freq(context):
        """
        cache the context frequency
        :param context: the context to cache
        :return: the frequency of specific context
        """
        if context not in context_cache:
            context_cache[context] = sum([lm.get(word).get(context, 0) for word in lm])
        return context_cache[context]

    sentence_proba = 1
    for i, word in enumerate(s_words):
        context = " ".join(s_words[max(0, i - n):i])
        seq_freq = lm.get(word, {"": 0}).get(context, 0)
        context_total_freq = _context_freq(context)
        sentence_proba *= float(seq_freq + 1) / (context_total_freq + V)
    return sentence_proba
