import unittest

import redis

from autocompleter import (
    DictStorageAutoCompleter,
    DictStorageSpellChecker,
    DictStorageTokenizer,
    RedisStorageAutoCompleter,
    RedisStorageSpellChecker,
    RedisStorageTokenizer,
    RequiresTraining,
)
redis_client = redis.from_url("http://localhost:6379")
process_cache = {}


class TestAutoCompleter(unittest.TestCase):

    def tearDown(self):
        super(TestAutoCompleter, self).tearDown()

        for cls in (DictStorageTokenizer,
                DictStorageSpellChecker):
            cls(process_cache).bust_cache()

        for cls in (RedisStorageTokenizer,
                    RedisStorageSpellChecker):
            cls(redis_client).bust_cache()

    def test_train_from_strings_token_to_full_string(self):
        """ Verifies that input training set is lowercased
        and divied up into N grams that map to tokens. """
        DictStorageTokenizer(process_cache).train_from_strings([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        strings_for_hello = DictStorageTokenizer(process_cache).get_full_strings_for_token("hello")
        self.assertEqual(strings_for_hello, set(['hello commrades', 'hello world']))

    def test_train_from_strings_n_gram_to_tokens(self):
        """ Verifies that n grams map to tokens. """
        DictStorageTokenizer(process_cache).train_from_strings([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        tokens_for_h = DictStorageTokenizer(process_cache).get_tokens_for_n_gram("h")
        self.assertEqual(tokens_for_h, set(['hey', 'hello']))

    def test_bust_cache(self):
        """ Verifies that the state of the class can be reset. """
        DictStorageTokenizer(process_cache).train_from_strings([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        DictStorageTokenizer(process_cache).bust_cache()
        with self.assertRaises(RequiresTraining):
            DictStorageTokenizer(process_cache).get_tokens_for_n_gram("g")

    def test_spell_check(self):
        """ Verifies that spell check works. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "hello world",
        ]
        DictStorageTokenizer(process_cache).train_from_strings(word_list)
        DictStorageSpellChecker(process_cache).train_from_strings(word_list)

        corrected_tokens = DictStorageSpellChecker(process_cache).correct_phrase("hye wrld")
        self.assertEqual(corrected_tokens, ["hey", "world"])

        corrected_tokens = DictStorageSpellChecker(process_cache).correct_phrase("CMMRADES")
        self.assertEqual(corrected_tokens, ["commrades"])

        corrected_tokens = DictStorageSpellChecker(process_cache).correct_phrase("hello there")
        self.assertEqual(corrected_tokens, ["hello", "there"])

    def test_autocomplete(self):
        """ Verifies that autocomplete works. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "today is a tremendous day",
        ]
        DictStorageTokenizer(process_cache).train_from_strings(word_list)
        guessed_phrases = DictStorageAutoCompleter(process_cache).guess_full_strings(["hello", "world"])
        self.assertEqual(
            guessed_phrases,
            ['hey there world', 'hello commrades']
        )
        guessed_phrases = DictStorageAutoCompleter(process_cache).guess_full_strings(["nothing", "in", "list"])
        self.assertEqual(guessed_phrases, [])

    def test_multiple_matches(self):
        """ Verifies that results are sorted by exact match with subsets of matches
        ranked last. """
        word_list = [
            "this will be repeated",
            "this will be repeated hardcore",
            "this will be repeated really hardcore",
        ]
        DictStorageTokenizer(process_cache).train_from_strings(word_list)
        guessed_phrases = DictStorageAutoCompleter(process_cache).guess_full_strings(
            ["this", "will", "be", "repeated", "hardcore"]
        )
        self.assertEqual(
            guessed_phrases,
            [
                'this will be repeated hardcore',
                'this will be repeated really hardcore',
                'this will be repeated'
            ]
        )

    def test_redis_tokenizer(self):
        """ Verifies that Redis storage maintains existing logic. """
        RedisStorageTokenizer(redis_client).train_from_strings([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        strings_for_hello = RedisStorageTokenizer(redis_client).get_full_strings_for_token("hello")
        self.assertEqual(strings_for_hello, set(['hello commrades', 'hello world']))

    def test_redis_spellchecker(self):
        """ Verifies that Redis storage maintains existing logic. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "hello world",
        ]
        RedisStorageTokenizer(redis_client).train_from_strings(word_list)
        RedisStorageSpellChecker(redis_client).train_from_strings(word_list)

        corrected_tokens = RedisStorageSpellChecker(redis_client).correct_phrase("hye wrld")
        self.assertEqual(corrected_tokens, ["hey", "world"])

    def test_redis_autocompleter(self):
        """ Verifies that Redis storage maintains existing logic. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "today is a tremendous day",
        ]
        RedisStorageTokenizer(redis_client).train_from_strings(word_list)
        guessed_phrases = RedisStorageAutoCompleter(redis_client).guess_full_strings(["hello", "world"])
        self.assertEqual(
            guessed_phrases,
            ['hey there world', 'hello commrades']
        )

    def test_integration(self):
        """ Verifies actual full use case of spellchecking and autocompleting. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "hello world",
        ]
        RedisStorageTokenizer(redis_client).train_from_strings(word_list)
        RedisStorageSpellChecker(redis_client).train_from_strings(word_list)
        corrected_tokens = RedisStorageSpellChecker(redis_client).correct_phrase("hye wrld")
        guessed_phrases = RedisStorageAutoCompleter(redis_client).guess_full_strings(corrected_tokens)
        self.assertEqual(
            guessed_phrases,
            ['hey there world', 'hello world']
        )

    def test_redis_requires_training(self):
        """ Verifies that exceptions are thrown for empty data in redis. """
        with self.assertRaises(RequiresTraining):
            RedisStorageSpellChecker(redis_client).correct_phrase("hye wrld")
        with self.assertRaises(RequiresTraining):
            RedisStorageAutoCompleter(redis_client).guess_full_strings(["hello"])

    def test_numeric_input(self):
        """ Verifies that numbers are ignored for tokenization. """
        word_list = [
            "I have 99 problems",
        ]
        RedisStorageTokenizer(redis_client).train_from_strings(word_list)
        RedisStorageSpellChecker(redis_client).train_from_strings(word_list)
        problems_count = RedisStorageSpellChecker(redis_client).get_count_for_token("problems")
        self.assertEqual(
            problems_count,
            1
        )
        corrected_tokens = RedisStorageSpellChecker(redis_client).correct_phrase("I have 99 problems")
        self.assertEqual(corrected_tokens, ['i', 'have', '99', 'problems'])

    def test_train_multiple(self):
        """ Verifies that training updates a model rather than re-trains it. """
        word_list = [
            "octopus",
        ]
        DictStorageAutoCompleter(process_cache).train_from_strings(word_list)

        word_list = [
            "rabbit",
        ]
        DictStorageAutoCompleter(process_cache).train_from_strings(word_list)

        corrected_tokens = DictStorageAutoCompleter(process_cache).correct_phrase("octipus rbbit")
        self.assertEqual(corrected_tokens, ["octopus", "rabbit"])

        guessed_phrases = DictStorageAutoCompleter(
            process_cache
        ).guess_full_strings(corrected_tokens)
        self.assertEqual(guessed_phrases, ['octopus', 'rabbit'])

    def test_train_multiple_redis(self):
        """ Verifies that training updates a model rather than re-trains it. """
        word_list = [
            "octopus",
        ]
        RedisStorageAutoCompleter(redis_client).train_from_strings(word_list)

        word_list = [
            "rabbit",
        ]
        RedisStorageAutoCompleter(redis_client).train_from_strings(word_list)

        corrected_tokens = RedisStorageAutoCompleter(redis_client).correct_phrase("octipus rbbit")
        self.assertEqual(corrected_tokens, ["octopus", "rabbit"])

        guessed_phrases = RedisStorageAutoCompleter(
            redis_client
        ).guess_full_strings(corrected_tokens)
        self.assertEqual(guessed_phrases, ['octopus', 'rabbit'])
