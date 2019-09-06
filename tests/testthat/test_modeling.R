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


test_that("The BertConfig routines work", {
  config <- BertConfig(vocab_size = 30522L)
  expected_config <- list("vocab_size" = 30522L,
                          "hidden_size" = 768L,
                          "num_hidden_layers" = 12L,
                          "num_attention_heads" = 12L,
                          "hidden_act" = "gelu",
                          "intermediate_size" = 3072L,
                          "hidden_dropout_prob" = 0.1,
                          "attention_probs_dropout_prob" = 0.1,
                          "max_position_embeddings" = 512L,
                          "type_vocab_size" = 16L,
                          "initializer_range" = 0.02)
  testthat::expect_is(config, "BertConfig")
  testthat::expect_identical(names(config), names(expected_config))

  json_file <- "bert_config.json"
  json_config <- bert_config_from_json_file(json_file)

  testthat::expect_is(json_config, "BertConfig")
  testthat::expect_identical(names(json_config), names(expected_config))
})

test_that("The BertModel routine works", {
  with(tensorflow::tf$variable_scope("tests",
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
                              num_attention_heads = 12L,
                              intermediate_size = 1024L)
         model_train <- BertModel(config = config,
                                  is_training = TRUE,
                                  input_ids = input_ids,
                                  input_mask = input_mask,
                                  token_type_ids = token_type_ids)
         model <- BertModel(config = config,
                            is_training = FALSE,
                            input_ids = input_ids,
                            input_mask = NULL,
                            token_type_ids = NULL)
       }
  )
  testthat::expect_is(model, "BertModel")
  testthat::expect_is(model$embedding_output,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(model$embedding_table,
                      "tensorflow.python.ops.variables.RefVariable")
  testthat::expect_is(model$sequence_output,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(model$pooled_output,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(model$all_encoder_layers[[1]],
                      "tensorflow.python.framework.ops.Tensor")

  # dropout should only be applied in training!
  testthat::expect_true(grepl(pattern = "dropout",
                              model_train$embedding_output$op$name))
  testthat::expect_false(grepl(pattern = "dropout",
                               model$embedding_output$op$name))
})


test_that("gelu works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       tfx <- tensorflow::tf$get_variable("tfx", tensorflow::shape(10L))
  )
  tgelu <- gelu(tfx)
  testthat::expect_is(tgelu, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_identical(tgelu$shape$as_list(), 10L)
})

test_that("get_activation works", {
  testthat::expect_identical(get_activation("gelu"), gelu)
  testthat::expect_equal(get_activation("relu"),
                         tensorflow::tf$nn$relu)
  testthat::expect_equal(get_activation("tanh"),
                         tensorflow::tf$tanh)
  testthat::expect_true(is.na(get_activation("linear")))
})

# test_that("get_assignment_map_from_checkpoint works", {
#   # Create a "model" with a couple variables that overlap some variable
#   # names in the BERT checkpoint. (The actual variables aren't compatible
#   # with the checkpoint.)
#   # The BERT checkpoint is large, and won't be included in repo. Eventually
#   # should save a tiny checkpoint to use for testing purposes. For now,
#   # run this test only if the checkpoint can be found.
#
#   init_checkpoint <- file.path("/shared",
#                                "BERT_checkpoints",
#                                "uncased_L-12_H-768_A-12",
#                                "bert_model.ckpt")
#
#   # Checkpoint "path" is actually only a stub filename; add ".index" to
#   # check for a specific file.
#   testthat::skip_if_not(file.exists(paste0(init_checkpoint,
#                                            ".index")),
#                         message = "Checkpoint index not found; skipping test.")
#
#   with(tensorflow::tf$variable_scope("bert",
#                                      reuse = tensorflow::tf$AUTO_REUSE ),
#        {
#          test_ten1 <- tensorflow::tf$get_variable(
#            "encoder/layer_9/output/dense/bias",
#            shape = c(1L, 2L, 3L)
#          )
#          test_ten2 <- tensorflow::tf$get_variable(
#            "encoder/layer_9/output/dense/kernel",
#            shape = c(1L, 2L, 3L)
#          )
#        }
#   )
#   tvars <- tensorflow::tf$get_collection(
#     tensorflow::tf$GraphKeys$GLOBAL_VARIABLES
#   )
#
#   amap <- get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#   expected_result <- readRDS("sample_amap.rds")
#   testthat::expect_identical(amap, expected_result)
# })


test_that("dropout works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       todrop <- tensorflow::tf$get_variable("todrop",
                                             tensorflow::shape(10L, 20L))
  )
  dropped <- dropout(todrop, 0.3)
  testthat::expect_is(dropped, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_true(grepl(pattern = "dropout", dropped$op$name))
})

test_that("layer_norm works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       lnorm <- tensorflow::tf$get_variable("lnorm", tensorflow::shape(10L))
  )
  normed <- layer_norm(lnorm)
  testthat::expect_is(normed, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_true(grepl(pattern = "LayerNorm", normed$op$name))
})

test_that("layer_norm_and_dropout works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       lndr <- tensorflow::tf$get_variable("lndr", tensorflow::shape(10L))
  )
  normed_and_dropped <- layer_norm_and_dropout(lndr, dropout_prob = 0.5)
  testthat::expect_is(normed_and_dropped,
                      "tensorflow.python.framework.ops.Tensor")
  testthat::expect_true(grepl(pattern = "dropout", normed_and_dropped$op$name))
})

test_that("create_initializer works", {
  init <- create_initializer()
  testthat::expect_is(init, "tensorflow.python.ops.init_ops.TruncatedNormal")
})

test_that("embedding_lookup works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         ids <- tensorflow::tf$get_variable("ids", dtype = "int32",
                                            shape = tensorflow::shape(10, 20))
         el <- embedding_lookup(ids, vocab_size = 100L,
                                word_embedding_name = "some_name")
       }
  )
  testthat::expect_is(el[[1]], "tensorflow.python.framework.ops.Tensor")
  testthat::expect_is(el[[2]], "tensorflow.python.ops.variables.RefVariable")
})

test_that("embedding_postprocessor works", {
  batch_size <- 10
  seq_length <- 512
  embedding_size <- 200
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         input_tensor <- tensorflow::tf$get_variable(
           "input_epp", dtype = "float",
           shape = tensorflow::shape(batch_size, seq_length, embedding_size))
         token_type_ids <- tensorflow::tf$get_variable(
           "ids_epp", dtype = "int32",
           shape = tensorflow::shape(batch_size, seq_length))

         pp_embed <- embedding_postprocessor(input_tensor,
                                             use_token_type = TRUE,
                                             token_type_ids = token_type_ids)
       }
  )
  testthat::expect_is(pp_embed, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_true(grepl(pattern = "dropout", pp_embed$op$name))
})

test_that("create_attention_mask_from_input_mask works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         from_tensor <- ids <- tensorflow::tf$get_variable(
           "ften",
           dtype = "float",
           shape = tensorflow::shape(10, 20)
         )
         to_mask <- ids <- tensorflow::tf$get_variable(
           "mask",
           dtype = "int32",
           shape = tensorflow::shape(10, 30)
         )
         amask <- create_attention_mask_from_input_mask(from_tensor, to_mask)
       }
  )
  testthat::expect_is(amask, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_identical(amask$shape$as_list(), c(10L, 20L, 30L))
})

test_that("transformer_model works", {
  batch_size <- 10
  seq_length <- 500
  hidden_size <- 120
  num_hidden <- 7

  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         input_tensor <- tensorflow::tf$get_variable("input_tm",
                                                     shape = c(batch_size,
                                                               seq_length,
                                                               hidden_size))
         model_t <- transformer_model(input_tensor = input_tensor,
                                      hidden_size = hidden_size,
                                      num_hidden_layers = num_hidden,
                                      do_return_all_layers = TRUE)
       }
  )
  # ATTN: modified below to account for attention_data
  attention_data <- model_t$attention_data
  testthat::expect_equal(length(attention_data), num_hidden)
  testthat::expect_is(attention_data[[num_hidden]],
                      "tensorflow.python.framework.ops.Tensor")
  model_t <- model_t$final_outputs
  # ATTN: modified above to account for attention_data

  testthat::expect_equal(length(model_t), num_hidden)
  testthat::expect_is(model_t[[num_hidden]],
                      "tensorflow.python.framework.ops.Tensor")
})


test_that("get_shape_list works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         phold <- tensorflow::tf$placeholder(tensorflow::tf$int32,
                                             shape = tensorflow::shape(4))
         static_shape <- get_shape_list(phold)
         tfunique <- tensorflow::tf$unique(phold)
         tfy <- tfunique$y
         dynamic_shape <- get_shape_list(tfy)
       }
  )
  testthat::expect_identical(static_shape, list(4L))
  testthat::expect_is(dynamic_shape[[1]],
                      "tensorflow.python.framework.ops.Tensor")
})

test_that("reshape to/from matrix functions work", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       r3t <- tensorflow::tf$get_variable("r3t", dtype = "int32",
                                          shape = tensorflow::shape(10, 20, 3))
  )
  mat <- reshape_to_matrix(r3t)
  testthat::expect_is(mat, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_identical(mat$shape$as_list(), c(200L, 3L))

  ten3 <- reshape_from_matrix(mat, orig_shape_list = list(10L, 20L, 3L))
  testthat::expect_is(ten3, "tensorflow.python.framework.ops.Tensor")
  testthat::expect_identical(ten3$shape$as_list(), c(10L, 20L, 3L))
})

test_that("assert_rank works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         ten <- tensorflow::tf$get_variable("ten", dtype = "int32",
                                            shape = tensorflow::shape(10))
         testthat::expect_true(assert_rank(ten, 1))
         testthat::expect_true(assert_rank(ten, 1:2))
         testthat::expect_error(assert_rank(ten, 2), "not equal")
       }
  )
})
