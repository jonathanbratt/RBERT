# RBERT 0.1.11

* Added parameter to shush verbose `extract_features`.
* Removed vestigal `use_one_hot_embeddings` parameter from everywhere.
* `extract_features` now can take plain character vectors as input.
* `extract_features` now can take a single checkpoint directory or model name,
rather than three separate file paths.

# RBERT 0.1.7

* Updated `extract_features` to return tidy tibbles (@jonthegeek, #29)

# RBERT 0.1.6

* Updated `download_BERT_checkpoint` to simplify usage. (@jonthegeek, #25)

# RBERT 0.1.0

* Added a `NEWS.md` file to track changes to the package.
* Initial open source release.
