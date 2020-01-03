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
  examples <- list(
    InputExample_EF(
      unique_id = 1,
      text_a = "I saw the branch on the bank.",
      text_b = "A second sequence of words."
    ),
    InputExample_EF(
      unique_id = 2,
      text_a = "I saw the branch of the bank."
    )
  )
  # tokenizer <- FullTokenizer("vocab.txt")
  # saveRDS(tokenizer, here::here("tests", "testthat", "tokenizer.rds"))
  # tokenizer <- readRDS(here::here("tests", "testthat", "tokenizer.rds"))
  tokenizer <- readRDS("tokenizer.rds")
  feat_in <- .convert_single_example_EF(
    ex_index = 6L,
    example = examples[[2]],
    seq_length = 5L,
    tokenizer = tokenizer
  )
  expected_feat_in <- readRDS("sample_feat_in.rds")
  testthat::expect_identical(feat_in, expected_feat_in)


  # Run these tests only if checkpoint is found.
  BERT_PRETRAINED_DIR <- cpdir # from setup.R

  # Test the ckpt_dir argument here. (Expect no error.)
  feats <- extract_features(
    examples = examples,
    ckpt_dir = BERT_PRETRAINED_DIR,
    batch_size = 2L
  )

  testthat::expect_error(
    extract_features(
      examples = examples,
      batch_size = 2L
    ),
    "ckpt_dir"
  )

  # We should get the same thing if we specify by model instead.
  feats2 <- extract_features(
    examples = examples,
    model = "bert_base_uncased",
    batch_size = 2L
  )

  expect_identical(feats2, feats)
  rm(feats2)

  # Also make sure it fails if they don't have the model.
  expect_error(
    extract_features(
      examples = examples,
      model = "bert_base_cased"
    ),
    "Specify ckpt_dir"
  )

  vocab_file <- file.path(BERT_PRETRAINED_DIR, "vocab.txt")
  init_checkpoint <- file.path(BERT_PRETRAINED_DIR, "bert_model.ckpt")
  # Checkpoint "path" is actually only a stub filename; add ".index" to
  # check for a specific file.
  testthat::skip_if_not(file.exists(paste0(
    init_checkpoint,
    ".index"
  )),
  message = "Checkpoint index not found; skipping test."
  )

  bert_config_file <- file.path(BERT_PRETRAINED_DIR, "bert_config.json")

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
    sort(unique(feats$output$token)),
    sort(unique(tokens))
  )

  # By default we fetch the last 4 layers.
  expect_equal(nrow(feats$output), length(tokens) * 4)
  expect_equal(
    ncol(feats$output),
    5L + 768L
  )
  expect_length(feats, 1)

  # Make sure we can grab layer 0 when we want to.
  feats <- extract_features(
    examples = examples,
    vocab_file = vocab_file,
    bert_config_file = bert_config_file,
    init_checkpoint = init_checkpoint,
    batch_size = 2L,
    layer_indexes = -4:0,
    features = c("output", "attention")
  )

  expect_length(feats, 2)

  # There may be some minor numerical differences across different systems. Need
  # to do a comparison along the lines of dplyr::near. Needed to update these
  # tests for the new format, because some of the layer/token index repeats went
  # away, and thus the sum changed. I tibbled the expected feats and resaved.
  test_feats_flat <- suppressWarnings(as.numeric(unlist(feats$output)))

  # expected_feats <- readRDS(
  #   here::here("tests", "testthat", "sample_feats.rds")
  # )
  expected_feats <- readRDS("sample_feats.rds")
  # The sorting changed since I saved an example, so let's put it into the same
  # order as the one we're getting now.
  expected_feats <- dplyr::arrange(
    expected_feats,
    sequence_index,
    layer_index,
    token_index
  )
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
  na.rm = TRUE
  )

  testthat::expect_lte(mean_relative_difference, tol)

  test_attn_flat <- suppressWarnings(
    as.numeric(unlist(feats$attention$attention_weight))
  )

  # expected_attn <- readRDS(
  #   here::here("tests", "testthat", "attention_probs.rds")
  # )
  expected_attn <- readRDS("attention_probs.rds")
  expected_attn_flat <- suppressWarnings(as.numeric(unlist(expected_attn)))
  expected_attn_flat <- expected_attn_flat[!is.na(expected_attn_flat)]

  # The original expected value has rotated matrices relative to the tidy
  # tibble. However, it's unlikely that they'd work out to have sum and mean
  # below within tolerance if they were actually different, so I'm sorting to
  # get a "good enough" evaluation.
  expected_attn_flat <- sort(expected_attn_flat)
  test_attn_flat <- sort(test_attn_flat)

  rel_diff_sum <- abs(sum(test_attn_flat, na.rm = TRUE) -
    sum(expected_attn_flat, na.rm = TRUE)) /
    (tol + abs(sum(test_attn_flat, na.rm = TRUE) +
      sum(expected_attn_flat, na.rm = TRUE)))
  testthat::expect_lte(rel_diff_sum, tol)

  mean_relative_difference <- mean(abs(test_attn_flat - expected_attn_flat) /
    (tol + abs(test_attn_flat +
      expected_attn_flat)),
  na.rm = TRUE
  )

  testthat::expect_lte(mean_relative_difference, tol)

  feats <- extract_features(
    examples = examples,
    vocab_file = vocab_file,
    bert_config_file = bert_config_file,
    init_checkpoint = init_checkpoint,
    batch_size = 2L,
    features = "output"
  )
  expect_length(feats, 1)
  expect_is(feats$output, "tbl_df")
  expect_equal(
    colnames(feats$output),
    c(
      "sequence_index", "segment_index", "token_index", "token", "layer_index",
      paste0("V", 1:768)
    )
  )

  feats <- extract_features(
    examples = examples,
    vocab_file = vocab_file,
    bert_config_file = bert_config_file,
    init_checkpoint = init_checkpoint,
    batch_size = 2L,
    features = "attention"
  )
  expect_length(feats, 1)
  expect_is(feats$attention, "tbl_df")
  expect_equal(
    colnames(feats$attention),
    c(
      "sequence_index", "token_index", "segment_index", "token",
      "layer_index", "head_index", "attention_token_index",
      "attention_segment_index", "attention_token", "attention_weight"
    )
  )

  # works for examples given as character vectors
  text_example1 <- "one"
  text_example2 <- list(c("one", "two"), c("three", "four"))
  text_example3 <- list(list("one", "two"), list("three", "four"))
  lone_example <- make_examples_simple(text_example1)[[1]]

  feats1 <- extract_features(
    examples = text_example1,
    model= "bert_base_uncased"
  )
  testthat::expect_equal(dim(feats1$output), c(12L, 773L))

  feats1b <- extract_features(
    examples = lone_example,
    model= "bert_base_uncased"
  )
  testthat::expect_identical(feats1, feats1b)

  feats2 <- extract_features(
    examples = text_example2,
    model= "bert_base_uncased"
  )
  testthat::expect_equal(dim(feats2$output), c(40L, 773L))

  feats3 <- extract_features(
    examples = text_example3,
    model= "bert_base_uncased"
  )
  testthat::expect_identical(feats2, feats3)

  # Manual speed tests:
  # emma_lines <- janeaustenr::emma[janeaustenr::emma != ""][5:54]
  # examples <- purrr::imap(
  #   emma_lines,
  #   ~ InputExample_EF(unique_id = .y, text_a = .x)
  # )
  # microbenchmark::microbenchmark(
  #   feats <- extract_features(
  #     examples = examples,
  #     vocab_file = vocab_file,
  #     bert_config_file = bert_config_file,
  #     init_checkpoint = init_checkpoint,
  #     batch_size = 2L,
  #     features = "attention"
  #   ),
  #   times = 1
  # )
})

test_that(".get_actual_index works", {
  testthat::expect_error(
    .get_actual_index(index = 0, length = 10),
    "Ambiguous"
  )

  testthat::expect_error(
    .get_actual_index(index = 11, length = 10),
    "out of range"
  )

  testthat::expect_identical(.get_actual_index(index = -2, length = 10), 9L)

  testthat::expect_identical(.get_actual_index(index = 9, length = 10), 9L)
})

test_that("make_examples_simple works", {
  text <- c(
    "Here are some words.",
    "Here are some more words."
  )
  input_ex <- make_examples_simple(text)
  testthat::expect_s3_class(input_ex[[1]], "InputExample_EF")

  testthat::expect_identical(input_ex[[1]]$text_a, text[[1]])
  testthat::expect_null(input_ex[[1]]$text_b)
  testthat::expect_identical(input_ex[[2]]$text_a, text[[2]])
  testthat::expect_null(input_ex[[2]]$text_b)
})

test_that("make_examples_simple works for two-segment examples", {
  text <- list(
    c(
      "First sequence, first segment.",
      "First sequence, second segment."
    ),
    c(
      "Second sequence, first segment.",
      "Second sequence, second segment.",
      "Second sequence, EXTRA segment."
    ),
    "Third sequence, only one segment."
  )
  testthat::expect_warning(
    input_ex <- make_examples_simple(text),
    "ignored"
  )
  testthat::expect_identical(input_ex[[1]]$text_a, text[[1]][[1]])
  testthat::expect_identical(input_ex[[1]]$text_b, text[[1]][[2]])
  testthat::expect_identical(input_ex[[2]]$text_a, text[[2]][[1]])
  testthat::expect_identical(input_ex[[2]]$text_b, text[[2]][[2]])
  testthat::expect_identical(input_ex[[3]]$text_a, text[[3]])
  testthat::expect_null(input_ex[[3]]$text_b)
})
