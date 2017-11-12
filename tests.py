from spell_checker import learn_language_model, create_error_distribution, _normalize_text, correct_sentence, \
    correct_word, evaluate_text
from collections import Counter

if __name__ == '__main__':
    lm = learn_language_model(["data/trump_historical_tweets.txt", "data/big.txt"], 3, None)
    error_dist = create_error_distribution("data/wikipedia_common_misspellings.txt")
    words = []
    with open("data/trump_historical_tweets.txt", "r") as f:
        for line in f:
            cleaned_line = _normalize_text(line)
            words = words + [x for x in cleaned_line.split(" ") if x != ""]
    word_freq = Counter(words)
    print(correct_word("idae", word_freq, error_dist))
    print(evaluate_text(
        "@Janetlarose1: @realDonaldTrump @FaceTheNation @jdickerson WASHINGTON VERSUS TRUMP  &TRUMPS SUPPORTERS ... #TRUMPDOG",
        # TODO choose different sentence
        lm))
    # for _ in range(5):
    #     print(generate_text(lm))
    print(correct_sentence("how aer you", lm, error_dist))
    print(correct_sentence("how aer yuo", lm, error_dist))
    print(correct_sentence("hw are you", lm, error_dist))
    pass
