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

# hmm. I think I need to use `create_layer` everywhere that I've been using
# something like layer_object(inputs), following python.
# Rather than:
# embedding_output <- self$embeddings_layer(inputs)
# I would need to define the wrapper function with
# `create_layer`, and then call *that*. Maybe?


#' @export
BertModelLayer <- R6::R6Class(
  "BertModelLayer",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  public = list(
    initialize = function(param_list = list(), ...) {
      self$params <- list(
        vocab_size = NULL,
        use_token_type = TRUE,
        use_position_embeddings = TRUE,
        token_type_vocab_size = 2L,
        embedding_size = NULL,   # NULL for BERT, not NULL for ALBERT
        max_position_embeddings = 512L,
        num_layers = NULL,
        out_layer_ndxs = NULL,   # [-1] # which layers to return. 1-indexed.
        shared_layer = FALSE,  # False for BERT, True for ALBERT
        intermediate_size  = NULL,
        intermediate_activation = "gelu",
        hidden_size = 768L,
        num_heads = NULL,
        hidden_dropout = 0.1,
        attention_dropout = 0.1,
        initializer_range = 0.02,
        size_per_head = NULL,
        query_activation = NULL,
        key_activation = NULL,
        value_activation = NULL,
        negative_infinity = -10000.0,
        trainable = TRUE,
        name = NULL,
        dtype = tensorflow::tf$float32$name,
        dynamic = FALSE
      )
      self$params <- .update_list(self$params, param_list)
      self$params <- .update_list(self$params, list(...))

      self$embeddings_layer <- NULL
      self$encoders_layer <- NULL

      self$support_masking <- TRUE
    },

    build = function(input_shape) {
      # (input_spec?)
      self$embeddings_layer <-  custom_layer_bert_embeddings(
        param_list = self$params,
        name = "embeddings"
      )

      self$encoders_layer <- custom_layer_tranformer_encoder(
        param_list = self$params,
        name = "encoder"
      )

      super$build(input_shape)
    },

    call = function(inputs, mask = NULL, training = NULL) {
      if (is.null(mask)) {
        mask <- self$embeddings_layer$compute_mask(inputs)
      }

      embedding_output <- self$embeddings_layer(inputs,
                                                mask = mask,
                                                training = training)
      output <- self$encoders_layer(embedding_output,
                                    mask = mask,
                                    training = training)
      return(output)   # [B, seq_len, hidden_size]
    }
  )
)

# wrapper function
#' @export
custom_layer_bert_model <- function(object,
                                    trainable = TRUE,
                                    param_list = list(),
                                    ...) {
  keras::create_layer(layer_class = BertModelLayer,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}
