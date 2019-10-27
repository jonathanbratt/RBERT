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

test_that("features and examples routines work", {
  examples <- list(InputExample_EF(unique_id = 1,
                                   text_a = "I saw the branch on the bank.",
                                   text_b = "A second sequence of words."),
                   InputExample_EF(unique_id = 2,
                                   text_a = "I saw the branch of the bank."))
  # tokenizer <- FullTokenizer("vocab.txt")
  # saveRDS(tokenizer, here::here("tests", "testthat", "tokenizer.rds"))
  tokenizer <- readRDS(here::here("tests", "testthat", "tokenizer.rds"))
  # tokenizer <- readRDS("tokenizer.rds")
  feat_in <- .convert_single_example_EF(ex_index = 6L,
                                    example = examples[[2]],
                                    seq_length = 5L,
                                    tokenizer = tokenizer)
  expected_feat_in <- readRDS("sample_feat_in.rds")
  testthat::expect_identical(feat_in, expected_feat_in)


  # Run these test only if checkpoint is found.
  BERT_PRETRAINED_DIR <- cpdir # from setup.R

  vocab_file <- file.path(BERT_PRETRAINED_DIR, 'vocab.txt')
  init_checkpoint <- file.path(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
  # Checkpoint "path" is actually only a stub filename; add ".index" to
  # check for a specific file.
  testthat::skip_if_not(file.exists(paste0(init_checkpoint,
                                           ".index")),
                        message = "Checkpoint index not found; skipping test.")

  bert_config_file <- file.path(BERT_PRETRAINED_DIR, 'bert_config.json')

  feats <- extract_features(examples = examples,
                            vocab_file = vocab_file,
                            bert_config_file = bert_config_file,
                            init_checkpoint = init_checkpoint,
                            batch_size = 2L)

  # Each token should be repeated 4 times (once for each of the 4 layers
  # requested by default). I'm sure there's a better way to do this, but this
  # works for these sentences.
  tokens <- unlist(
    c(
      "[CLS]",
      tolower(stringr::str_extract_all(
        examples[[1]]$text_a,
        "\\b[^\\s]+\\b",
        simplify = TRUE
      )),
      ".",
      "[SEP]",
      tolower(stringr::str_extract_all(
        examples[[1]]$text_b,
        "\\b[^\\s]+\\b",
        simplify = TRUE
      )),
      ".",
      "[SEP]",
      "[CLS]",
      tolower(stringr::str_extract_all(
        examples[[1]]$text_a,
        "\\b[^\\s]+\\b",
        simplify = TRUE
      )),
      ".",
      "[SEP]"
    )
  )

  expect_equal(
    sort(unique(feats$layer_outputs$token)),
    sort(unique(tokens))
  )

  # By default we fetch the last 4 layers.
  expect_equal(nrow(feats$layer_outputs), length(tokens) * 4)
  expect_equal(
    ncol(feats$layer_outputs),
    5L + 768L
  )

  # Make sure we can grab layer 0 when we want to.
  feats <- extract_features(examples = examples,
                            vocab_file = vocab_file,
                            bert_config_file = bert_config_file,
                            init_checkpoint = init_checkpoint,
                            batch_size = 2L,
                            layer_indexes = -4:0)

  # There may be some minor numerical differences across different systems. Need
  # to do a comparison along the lines of dplyr::near. Needed to update these
  # tests for the new format, because some of the layer/token index repeats went
  # away, and thus the sum changed. I tibbled the expected feats and resaved.
  test_feats_flat <- suppressWarnings(as.numeric(unlist(feats$layer_outputs)))

  expected_feats <- readRDS("sample_feats.rds")
  expected_feats_flat <- suppressWarnings(as.numeric(unlist(expected_feats)))

  tol <- 10^(-5)

  # check both the sum and mean relative difference
  rel_diff_sum <- abs(sum(test_feats_flat, na.rm = TRUE) -
                        sum(expected_feats_flat, na.rm = TRUE)) /
    (tol + abs(sum(test_feats_flat, na.rm = TRUE) +
                 sum(expected_feats_flat, na.rm = TRUE)))
  testthat::expect_lte(rel_diff_sum, tol)

  mean_relative_difference <- mean(abs(test_feats_flat - expected_feats_flat) /
                                     (tol + abs(test_feats_flat +
                                                  expected_feats_flat)),
                                   na.rm = TRUE)

  testthat::expect_lte(mean_relative_difference, tol)

  test_attn_flat <- suppressWarnings(as.numeric(unlist(feats$attention_probs)))

  expected_attn <- readRDS("attention_probs.rds")
  expected_attn_flat <- suppressWarnings(as.numeric(unlist(expected_attn)))

  rel_diff_sum <- abs(sum(test_attn_flat, na.rm = TRUE) -
                        sum(expected_attn_flat, na.rm = TRUE)) /
    (tol + abs(sum(test_attn_flat, na.rm = TRUE) +
                 sum(expected_attn_flat, na.rm = TRUE)))
  testthat::expect_lte(rel_diff_sum, tol)

  mean_relative_difference <- mean(abs(test_attn_flat - expected_attn_flat) /
                                     (tol + abs(test_attn_flat +
                                                  expected_attn_flat)),
                                   na.rm = TRUE)

  testthat::expect_lte(mean_relative_difference, tol)
})


test_that(".get_actual_index works", {
  testthat::expect_error(.get_actual_index(index = 0, length = 10),
                         "Ambiguous")

  testthat::expect_error(.get_actual_index(index = 11, length = 10),
                         "out of range")

  testthat::expect_identical(.get_actual_index(index = -2, length = 10), 9L)

  testthat::expect_identical(.get_actual_index(index = 9, length = 10), 9L)
})

test_that("make_examples_simple works", {
  text <- c("Here are some words.",
            "Here are some more words.")
  input_ex <- make_examples_simple(text)
  testthat::expect_s3_class(input_ex[[1]], "InputExample_EF")

  testthat::expect_identical(input_ex[[1]]$text_a, text[[1]])
  testthat::expect_null(input_ex[[1]]$text_b)
  testthat::expect_identical(input_ex[[2]]$text_a, text[[2]])
  testthat::expect_null(input_ex[[2]]$text_b)
})

test_that("make_examples_simple works for two-segment examples", {
  text <- list(c("First sequence, first segment.",
                 "First sequence, second segment."),
               c("Second sequence, first segment.",
                 "Second sequence, second segment.",
                 "Second sequence, EXTRA segment."),
               "Third sequence, only one segment.")
  testthat::expect_warning(input_ex <- make_examples_simple(text),
                           "ignored")
  testthat::expect_identical(input_ex[[1]]$text_a, text[[1]][[1]])
  testthat::expect_identical(input_ex[[1]]$text_b, text[[1]][[2]])
  testthat::expect_identical(input_ex[[2]]$text_a, text[[2]][[1]])
  testthat::expect_identical(input_ex[[2]]$text_b, text[[2]][[2]])
  testthat::expect_identical(input_ex[[3]]$text_a, text[[3]])
  testthat::expect_null(input_ex[[3]]$text_b)
})
