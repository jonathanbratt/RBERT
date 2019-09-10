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
  tokenizer <- FullTokenizer("vocab.txt")
  feat_in <- convert_single_example_EF(ex_index = 6L,
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

  # more incremental tests for troubleshooting
  bert_config <-  bert_config_from_json_file(bert_config_file)
  testthat::expect_equal(length(bert_config), 11L)

  model_fn <- model_fn_builder_EF(
    bert_config = bert_config,
    init_checkpoint = init_checkpoint,
    layer_indexes = 1:12,
    use_tpu = FALSE,
    use_one_hot_embeddings = FALSE)
  testthat::expect_is(model_fn, "function")

  is_per_host <- tensorflow::tf$contrib$tpu$InputPipelineConfig$PER_HOST_V2
  run_config <- tensorflow::tf$contrib$tpu$RunConfig(
    master = NULL, # assume for now *not* for TPU
    tpu_config = tensorflow::tf$contrib$tpu$TPUConfig(
      num_shards = 8L,
      per_host_input_for_training = is_per_host))

  estimator <- tensorflow::tf$contrib$tpu$TPUEstimator(
    use_tpu=FALSE, # no tpu for now
    model_fn = reticulate::py_func(model_fn),
    config = run_config,
    predict_batch_size = 2L)
  testthat::expect_is(estimator,
                      "tensorflow.python.estimator.estimator.Estimator")


  features <-  convert_examples_to_features_EF(examples = examples,
                                               seq_length = 128L,
                                               tokenizer = tokenizer)
  input_fn <- input_fn_builder_EF(features = features,
                                  seq_length = 128L)

  result_iterator <- estimator$predict(reticulate::py_func(input_fn),
                                       yield_single_examples = TRUE)
  testthat::expect_is(result_iterator, "python.builtin.iterator")
  print(names(result_iterator))
  if ("next" %in% names(result_iterator)) {
    result <- result_iterator$`next`()
  } else {
    result <- result_iterator$`__next__`()
  }

  # result <- result_iterator$`next`()
  testthat::expect_equal(length(result), 26L)

  feats <- extract_features(examples = examples,
                            vocab_file = vocab_file,
                            bert_config_file = bert_config_file,
                            init_checkpoint = init_checkpoint,
                            batch_size = 2L)
  # testthat::expect_equal(length(feats$layer_outputs$example_1$features), 17L)
  # expected_feats <- readRDS("sample_feats.rds")
  # testthat::expect_identical(feats$layer_outputs, expected_feats)
  # expected_attention_probs <- readRDS("attention_probs.rds")
  # testthat::expect_identical(feats$attention_probs, expected_attention_probs)
})


test_that("get_actual_index works", {
  testthat::expect_error(get_actual_index(index = 0, length = 10),
                         "Ambiguous")

  testthat::expect_error(get_actual_index(index = 11, length = 10),
                         "out of range")

  testthat::expect_identical(get_actual_index(index = -2, length = 10), 9L)

  testthat::expect_identical(get_actual_index(index = 9, length = 10), 9L)
})

test_that("make_examples_simple works", {
  text <- c("Here are some words.",
            "Here are some more words.")
  input_ex <- make_examples_simple(text)
  testthat::expect_s3_class(input_ex[[1]], "InputExample_EF")
})
