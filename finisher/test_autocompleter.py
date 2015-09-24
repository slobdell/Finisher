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

    def test_build_tokens_from_input_token_to_full_string(self):
        """ Verifies that input training set is lowercased
        and divied up into N grams that map to tokens. """
        DictStorageTokenizer(process_cache).build_tokens_from_input([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        token_to_full_string = DictStorageTokenizer(process_cache).token_to_full_string
        self.assertEqual(len(token_to_full_string), 5)
        mapped_strings = set.union(*token_to_full_string.values())
        self.assertEqual(
            mapped_strings,
            set(['hey there world', 'hello commrades', 'hello world'])
        )

    def test_build_tokens_from_input_n_gram_to_tokens(self):
        """ Verifies that n grams map to tokens. """
        DictStorageTokenizer(process_cache).build_tokens_from_input([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        n_gram_to_tokens = DictStorageTokenizer(process_cache).n_gram_to_tokens
        self.assertEqual(len(n_gram_to_tokens), 25)
        mapped_values = set.union(*n_gram_to_tokens.values())
        self.assertEqual(
            mapped_values,
            set(['commrades', 'world', 'there', 'hello', 'hey'])
        )

    def test_bust_cache(self):
        """ Verifies that the state of the class can be reset. """
        DictStorageTokenizer(process_cache).build_tokens_from_input([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        DictStorageTokenizer(process_cache).bust_cache()
        with self.assertRaises(RequiresTraining):
            DictStorageTokenizer(process_cache).n_gram_to_tokens

    def test_spell_check(self):
        """ Verifies that spell check works. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "hello world",
        ]
        DictStorageTokenizer(process_cache).build_tokens_from_input(word_list)
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
        DictStorageTokenizer(process_cache).build_tokens_from_input(word_list)
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
        DictStorageTokenizer(process_cache).build_tokens_from_input(word_list)
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
        RedisStorageTokenizer(redis_client).build_tokens_from_input([
            "hey there world",
            "hello Commrades",
            "hello world",
        ])
        token_to_full_string = RedisStorageTokenizer(redis_client).token_to_full_string
        self.assertEqual(len(token_to_full_string), 5)

    def test_redis_spellchecker(self):
        """ Verifies that Redis storage maintains existing logic. """
        word_list = [
            "hey there world",
            "hello Commrades",
            "hello world",
        ]
        RedisStorageTokenizer(redis_client).build_tokens_from_input(word_list)
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
        RedisStorageTokenizer(redis_client).build_tokens_from_input(word_list)
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
        RedisStorageTokenizer(redis_client).build_tokens_from_input(word_list)
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
        RedisStorageTokenizer(redis_client).build_tokens_from_input(word_list)
        RedisStorageSpellChecker(redis_client).train_from_strings(word_list)
        token_to_count = RedisStorageSpellChecker(redis_client).token_to_count
        self.assertEqual(
            token_to_count,
            {'i': '1', 'problems': '1', 'have': '1'}
        )
        corrected_tokens = RedisStorageSpellChecker(redis_client).correct_phrase("I have 99 problems")
        self.assertEqual(corrected_tokens, ['i', 'have', '99', 'problems'])
