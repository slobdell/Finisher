Finisher is a lightweight autocompletion library for Python.  It can be used in situations where you do not want to add additional dependencies such as SOLR or Cloudsearch to provide autocompletion functionality.

# Installation

```python
pip install finisher
```

## Up Front Example

```python
import json

from finisher import DictStorageAutoCompleter


# This can be anything
with open("movie_titles.json", "rb") as f:
    movie_titles_json = f.read()
    movie_titles = json.loads(movie_titles_json)

arbitrary_cache = {}
autocompleter = DictStorageAutoCompleter(arbitrary_cache)
autocompleter.train_from_strings(movie_titles)
corrected_tokens = autocompleter.correct_phrase("big lebewski")
guessed_phrases = autocompleter.guess_full_strings(corrected_tokens)
'''
    Results are:
        [
            "big lebowski (usa)",
            "big, bigger, and biggest trucks and diggers (book w/ dvd)",
            "build it bigger: biggest warship",
            "big foot (a.k.a. bigfoot)",
            "bigger than big (sumo mack)",
            "bigger than big (activate)",
            "big 3",
            "biggest fan",
            "big lebowski (universal/ widescreen)",
            "build it bigger: big easy rebuild"
        ]
'''
```

## Practical Example

The above example uses a dictionary to cache data since recomputing mappings will clearly be expensive.  This works in a single process or on a web server running a single worker, but in practical terms you will need something like Redis to allow for storage between processes.
Redis example is below:

```python
from finisher import RedisStorageAutoCompleter
import redis

redis_client = redis.from_url("http://localhost:6379")
autocompleter = RedisStorageAutoCompleter(redis_client)
autocompleter.train_from_strings(movie_titles)
corrected_tokens = autocompleter.correct_phrase("big lebewski")
guessed_phrases = autocompleter.guess_full_strings(corrected_tokens)
```

## Training a model

A model can be updated incrementally:

```python

arbitrary_cache = {}

autocompleter = DictStorageAutoCompleter(arbitrary_cache)
autocompleter.train_from_strings(movie_titles[:1000])

autocompleter = DictStorageAutoCompleter(arbitrary_cache)
autocompleter.train_from_strings(movie_titles[1000:])
```

And if we need to re-train the model you could either:

```python
autocompleter.bust_cache()
```

or just change the variable used for caching.

## Spellcheck

It should be noted that spellcheck is supported as well.  Consistent with the above example:

```python
autocompleter = RedisStorageAutoCompleter(redis_client)
corrected_tokens = autocompleter.correct_phrase(
    "my fvaorite moive is expandables bcause schwarzanagger is cool"
)
'''
    output:
        [
            'my',
            'favorite',
            'moive',
            'is',
            'expendables',
            'cause',
            'schwarzenegger',
            'is',
            'cool'
        ]
'''
```
