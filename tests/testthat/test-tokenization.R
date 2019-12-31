# Copyright 2019 Bedford Freeman & Worth Pub Grp LLC DBA Macmillan Learning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


test_that("the convert token/id functions work", {
  vocab <- c("token1"=0, "token2"=1, "token3"=2)
  inv_voc <- names(vocab)
  names(inv_voc) <- vocab

  test_result <- convert_tokens_to_ids(vocab, c("token1", "token3"))
  expected_result <-  c("token1"=0, "token3"=2)
  testthat::expect_identical(test_result, expected_result)

  test_result <- convert_ids_to_tokens(inv_voc, c(1, 3))
  expected_result <-  c("0"="token1", "2"="token3")
  testthat::expect_identical(test_result, expected_result)
})

test_that("The FullTokenizer tokenizer works as expected", {
  f_tokenizer <- FullTokenizer("vocab.txt", TRUE)
  text <- "\u535A\u00E7anned words; tihs is Silly"
  test_result <- tokenize(f_tokenizer, text)
  expected_result <- c("\u535A", "canned", "words", ";",
                       "ti", "##hs", "is", "silly")
  testthat::expect_identical(test_result, expected_result)

  f_tokenizer <- FullTokenizer("vocab_small.txt", TRUE)
  text <- "know the unknowable!"
  test_result <- tokenize(f_tokenizer, text)
  expected_result <- c("know", "the", "un", "##know",
                       "##able", "[UNK]")
  testthat::expect_identical(test_result, expected_result)
})


test_that("Tokenizers handle edge cases correctly", {
  test_string <- "remove char: \ufffd "
  vocab <- load_vocab(vocab_file = "vocab.txt")

  b_tokenizer <- BasicTokenizer(TRUE)
  test_result <- tokenize(b_tokenizer, text = test_string)
  expected_result <- c("remove", "char", ":")
  testthat::expect_identical(test_result, expected_result)

  wp_tokenizer <- WordpieceTokenizer(vocab, max_input_chars_per_word = 4)
  test_result <- tokenize(wp_tokenizer, text = "excessively long")
  expected_result <- c("[UNK]", "long")
  testthat::expect_identical(test_result, expected_result)
})


test_that("whitespace_tokenize splits a string on whitespace", {
  test_string <- " some\ttext\nwith whitespace "
  test_result <- whitespace_tokenize(test_string)
  expected_result <- c("some", "text", "with", "whitespace")
  testthat::expect_identical(test_result, expected_result)
})



test_that("strip_accents replaces accented chars with nearest equivalents", {
  test_string <- "fa\u00E7ile"
  test_result <- strip_accents(test_string)
  expected_result <- "facile"
  testthat::expect_identical(test_result, expected_result)
})


test_that("split_on_punc splits a string before and after punctuation chars", {
  test_string <- "stop! don't touch that."
  test_result <- split_on_punc(test_string)
  expected_result <- c("stop", "!", " don", "'", "t touch that", ".")
  testthat::expect_identical(test_result, expected_result)

  test_string <- "!"
  test_result <- split_on_punc(test_string)
  expected_result <- c("!")
  testthat::expect_identical(test_result, expected_result)
})




test_that("is_whitespace correctly classifies characters", {
  # tests from BERT: tokenization_test.py
  testthat::expect_true(is_whitespace(" "))
  testthat::expect_true(is_whitespace("\t"))
  testthat::expect_true(is_whitespace("\r"))
  testthat::expect_true(is_whitespace("\n"))
  testthat::expect_true(is_whitespace("\u00A0")) # non-breaking space

  testthat::expect_false(is_whitespace("A"))
  testthat::expect_false(is_whitespace("-"))
})


test_that("is_control correctly classifies characters", {
  # tests from BERT: tokenization_test.py
  testthat::expect_true(is_control("\u0005")) # 'Enquiry' control character

  testthat::expect_false(is_control("A"))
  testthat::expect_false(is_control(" "))
  testthat::expect_false(is_control("\t"))
  testthat::expect_false(is_control("\r"))
})



test_that("is_punctuation correctly classifies characters", {
  # tests from BERT: tokenization_test.py
  testthat::expect_true(is_punctuation("-"))
  testthat::expect_true(is_punctuation("$"))
  testthat::expect_true(is_punctuation("`"))
  testthat::expect_true(is_punctuation("."))

  testthat::expect_false(is_punctuation("A"))
  testthat::expect_false(is_punctuation(" "))
})


test_that("tokenize_text works correctly", {
  text <- c("Who doesn't like tacos?", "Not me!")
  tokens <- tokenize_text(text = text, ckpt_dir = cpdir)
  testthat::expect_identical(length(tokens[[1]]), 10L)
  testthat::expect_identical(length(tokens[[2]]), 5L)
})

test_that("check_vocab works correctly", {
  to_check <- c("apple", "appl")
  vcheck <- check_vocab(words = to_check, ckpt_dir = cpdir)
  testthat::expect_identical(vcheck, c(TRUE, FALSE))
})
