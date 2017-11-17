from spell_checker import learn_language_model, create_error_distribution, _normalize_text, correct_sentence, \
    correct_word, evaluate_text
from collections import Counter

if __name__ == '__main__':
    # lm = learn_language_model(["data/big.txt"], 3, None)
    lexicon = learn_language_model(["data/error_dist_test_text.txt"], 1, None)
    error_dist = create_error_distribution("data/error_dist_test.txt",lexicon)
    print(correct_word("idae", lexicon, error_dist))
    # print(evaluate_text(
    #     "@Janetlarose1: @realDonaldTrump @FaceTheNation @jdickerson WASHINGTON VERSUS TRUMP  &TRUMPS SUPPORTERS ... #TRUMPDOG",
    #     # TODO choose different sentence
    #     lm))
    # for _ in range(5):
    #     print(generate_text(lm))
    print(correct_sentence("how aer you", lm, error_dist))
    print(correct_sentence("how aer yuo", lm, error_dist))
    print(correct_sentence("hw are you", lm, error_dist))
    pass
