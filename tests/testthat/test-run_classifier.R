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


test_that("truncate_seq_pair works", {
  tokens_a <- c("a", "b", "c", "d")
  tokens_b <- c("w", "x", "y", "z")
  trunc_seq <- truncate_seq_pair(tokens_a, tokens_b, 5)
  expected_result <- list(trunc_a = c("a", "b", "c"),
                          trunc_b = c("w", "x"))
  testthat::expect_identical(trunc_seq, expected_result)
})

test_that("create_model works", {
  with(tensorflow::tf$variable_scope("tests_class1",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         input_ids <- tensorflow::tf$constant(list(list(31L, 51L, 99L),
                                                   list(15L, 5L, 0L)))

         input_mask <- tensorflow::tf$constant(list(list(1L, 1L, 1L),
                                                    list(1L, 1L, 0L)))
         token_type_ids <- tensorflow::tf$constant(list(list(0L, 0L, 1L),
                                                        list(0L, 2L, 0L)))
         config <- BertConfig(vocab_size = 32000L,
                              hidden_size = 768L,
                              num_hidden_layers = 8L,
                              type_vocab_size = 2L,
                              num_attention_heads = 12L,
                              intermediate_size = 1024L)
         class_model <- create_model(bert_config = config,
                                     is_training = TRUE,
                                     input_ids = input_ids,
                                     input_mask = input_mask,
                                     segment_ids = token_type_ids,
                                     labels = c(1L, 2L),
                                     num_labels = 2L,
                                     use_one_hot_embeddings = FALSE)
       }
  )
  testthat::expect_is(class_model$loss,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(class_model$per_example_loss,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(class_model$logits,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(class_model$probabilities,
                      "tensorflow.python.framework.ops.Tensor")

  testthat::expect_true(grepl(pattern = "Mean",
                              class_model$loss$op$name))
  testthat::expect_true(grepl(pattern = "Neg",
                              class_model$per_example_loss$op$name))
  testthat::expect_true(grepl(pattern = "BiasAdd",
                              class_model$logits$op$name))
  testthat::expect_true(grepl(pattern = "Softmax",
                              class_model$probabilities$op$name))
})

test_that("model_fn_builder works", {
  # Run this test only if the checkpoint can be found.
  init_checkpoint <- file.path(cpdir,  # from setup.R
                               "bert_model.ckpt")

  # Checkpoint "path" is actually only a stub filename; add ".index" to
  # check for a specific file.
  testthat::skip_if_not(file.exists(paste0(init_checkpoint,
                                           ".index")),
                        message = "Checkpoint index not found; skipping test.")
  with(tensorflow::tf$variable_scope("tests_class2",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         input_ids <- tensorflow::tf$constant(list(list(31L, 51L, 99L),
                                                   list(15L, 5L, 0L)))

         input_mask <- tensorflow::tf$constant(list(list(1L, 1L, 1L),
                                                    list(1L, 1L, 0L)))
         token_type_ids <- tensorflow::tf$constant(list(list(0L, 0L, 1L),
                                                        list(0L, 2L, 0L)))
         config <- BertConfig(vocab_size = 30522L,
                              hidden_size = 768L,
                              num_hidden_layers = 8L,
                              type_vocab_size = 2L,
                              num_attention_heads = 12L,
                              intermediate_size = 3072L)

         test_mod_fn <- model_fn_builder(bert_config = config,
                                         num_labels = 2L,
                                         init_checkpoint = init_checkpoint,
                                         learning_rate = 0.01,
                                         num_train_steps = 20L,
                                         num_warmup_steps = 10L,
                                         use_tpu = FALSE,
                                         use_one_hot_embeddings = FALSE)
         # After we implement InputFeatures class, come back and add tests for
         # `test_mod_fn`. Something like this, but better:
         # features <- list()
         # features$input_ids <- input_ids
         # features$input_mask <- input_mask
         # features$segment_ids <- token_type_ids
         # features$label_ids <- c(1L, 2L)

         # mod_fn_output <- test_mod_fn(features = features,
         #             labels = NULL,
         #             mode = "train",
         #             params = NULL)
       }
  )
  # This isn't much of a test, but it does confirm that the maker function
  # ran, which is non-trivial.
  testthat::expect_is(test_mod_fn, "function")

})

test_that("Examples/features creation routines work", {
  tokenizer <- FullTokenizer("vocab.txt")
  input_ex1 <- InputExample(guid = 1L,
                            text_a = "Some text to classify.",
                            text_b = "More wordy words.",
                            label = "good")

  testthat::expect_is(input_ex1, "InputExample")
  testthat::expect_identical(names(input_ex1),
                             c("guid", "text_a", "text_b", "label"))
  input_ex2 <- InputExample(guid = 2L,
                            text_a = "This is another example.",
                            text_b = "So many words.",
                            label = "bad")
  feat <- convert_examples_to_features(examples = list(input_ex1, input_ex2),
                                       label_list = c("good", "bad"),
                                       max_seq_length = 15L,
                                       tokenizer = tokenizer)
  testthat::expect_identical(length(feat), 2L)
  testthat::expect_is(feat[[1]], "InputFeatures")
  testthat::expect_identical(names(feat[[1]]),
                             c("input_ids",
                               "input_mask",
                               "segment_ids",
                               "label_id",
                               "is_real_example"))
})
