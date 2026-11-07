from majjatantu.failure_detection import token_repetition_ratio


def test_token_repetition_ratio_detects_loops():
    # Highly repetitive sequence
    looped = ([1, 2, 3, 4, 5, 6, 7, 8] * 20)
    assert token_repetition_ratio(looped, ngram=8) > 0.5

    unique = list(range(100))
    assert token_repetition_ratio(unique, ngram=8) == 0.0
