import json

from finisher import DictStorageAutoCompleter
from finisher import RedisStorageAutoCompleter
import redis


if __name__ == "__main__":

    # This can be anything
    with open("movie_titles.json", "rb") as f:
        movie_titles_json = f.read()
        movie_titles = json.loads(movie_titles_json)

    # DICTIONARY EXAMPLE
    arbitrary_cache = {}
    autocompleter = DictStorageAutoCompleter(arbitrary_cache, min_n_gram_size=3)
    autocompleter.train_from_strings(movie_titles)

    corrected_tokens = autocompleter.correct_phrase("shcwarzenagger terminator kidergarden cop")
    guessed_phrases = autocompleter.guess_full_strings(corrected_tokens)

    # REDIS EXAMPLE
    redis_client = redis.from_url("http://localhost:6379")

    autocompleter = RedisStorageAutoCompleter(redis_client)
    autocompleter.train_from_strings(movie_titles)
    corrected_tokens = autocompleter.correct_phrase("expandables schwarzanagger")
    guessed_phrases = autocompleter.guess_full_strings(corrected_tokens)
