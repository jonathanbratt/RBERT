# Copyright 2020 Bedford Freeman & Worth Pub Grp LLC DBA Macmillan Learning.
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


# custom layer: position_embedding ----------------------------------------

#' @keywords internal
.custom_layer_position_embedding_init <- function(param_list = list(), ...) {
  # Quick way of mimicking the params-flow structure.
  self$params <- list(
    max_position_embeddings = 512L,
    hidden_size = 128L,
    initializer_range = 0.02,
    trainable = TRUE,
    name = "PositionEmbedding",
    dtype = tensorflow::tf$float32$name,
    dynamic = FALSE
  )
  self$params <- .update_list(self$params, param_list)
  self$params <- .update_list(self$params, list(...))

  self$embedding_table <- NULL

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_position_embedding_build <- function(input_shape) {
  self$embedding_table <- self$add_weight(
    name = "position_embedding",
    shape = list(self$params$max_position_embeddings,
                 self$params$hidden_size),
    initializer = keras::initializer_truncated_normal(
      stddev = self$params$initializer_range
    )
  )

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_position_embedding_call <- function(inputs,
                                             mask = NULL,
                                             training = NULL) {
  # Just return the embedding after verifying that seq_len is always less than
  # max_position_embeddings. There's no other dependence on inputs, as this
  # is just the embedding for the position index, which is already implicit in
  # the embedding matrix.
  seq_len <- inputs

  assert_op <- tensorflow::tf$debugging$assert_less_equal(
    seq_len,
    self$params$max_position_embeddings
  )
  with(tensorflow::tf$control_dependencies(list(assert_op)),
       {
         # slice to seq_len
         full_position_embeddings <- tensorflow::tf$slice(
           self$embedding_table,
           begin = list(0L, 0L),
           size = list(seq_len, -1L)
         )
       }
  )
  return(full_position_embeddings)
}

#' @keywords internal
.make_custom_layer_position_embedding <- function() {
  layer_function <- keras::Layer(
    classname = "PositionEmbedding",

    initialize = .custom_layer_position_embedding_init,
    build = .custom_layer_position_embedding_build,
    call = .custom_layer_position_embedding_call
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Position Embedding
#'
#' Create position embeddings for the input layer.
#'
#' @inheritParams custom_layer_layernorm
#' @param param_list A named list of parameter values used in defining the
#'   layer.
#'   \describe{
#'   \item{`hidden_size`}{Integer; The size of the output. Should match the
#'     size of the second input.}
#'   \item{`max_position_embeddings`}{Numeric; .}
#'   \item{`initializer_range`}{Numeric; the value passed in as the `stddev`
#'     parameter to the `initializer_truncated_normal` in the dense layer
#'     initializer.}
#'   \item{`dtype`}{The data type of the layer output. Defaults to "float32".
#'   Valid values from `tensorflow::tf$float32$name`, etc. }
#'  }
#'
#' @export
#' @md
custom_layer_position_embedding <- function(object,
                                       name = NULL,
                                       trainable = NULL,
                                       param_list = list(),
                                       ...) {
  keras::create_layer(layer_class = .custom_layers$position_embedding,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}


# custom layer: bert_embeddings -------------------------------------------


#' @keywords internal
.custom_layer_bert_embeddings_init <- function(param_list = list(), ...) {
  # Quick way of mimicking the params-flow structure.
  self$params <- list(
    vocab_size = NULL,
    use_token_type = TRUE,
    use_position_embeddings = TRUE,
    token_type_vocab_size = 2L,
    hidden_dropout = 0.1,
    hidden_size = 768L,
    #TODO: I think NA, not NULL, is the way to go here. Ugh, still need to fix
    # more. Have to change the param loader function if I keep NAs. Revisit
    # this question.
    embedding_size = NULL,   # NULL for BERT, not NULL for ALBERT.
    max_position_embeddings = 512L,
    initializer_range = 0.02,
    trainable = TRUE,
    name = NULL,
    dtype = tensorflow::tf$float32$name,
    dynamic = FALSE
  )
  self$params <- .update_list(self$params, param_list)
  self$params <- .update_list(self$params, list(...))

  #TODO: do I really need to declare these all here?
  self$word_embeddings_layer <- NULL
  self$token_type_embeddings_layer <- NULL
  self$position_embeddings_layer <- NULL
  self$word_embeddings_projector_layer <- NULL  # for ALBERT
  self$layer_norm <- NULL
  self$dropout_layer <- NULL

  self$supports_masking <- TRUE


  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_bert_embeddings_build <- function(input_shape) {
  # use either hidden_size for BERT or embedding_size for ALBERT
  # (one of the features of ALBERT is that the embedding size doesn't
  # *have* to be the hidden size.)
  if (is.null(self$params$embedding_size)) {
    #BERT
    position_embedding_size <- self$params$hidden_size
  } else {
    #ALBERT
    position_embedding_size <- self$params$embedding_size
  }

  self$word_embeddings_layer <- keras::layer_embedding(
    input_dim = self$params$vocab_size,
    output_dim = position_embedding_size,
    mask_zero = TRUE,
    name = "word_embeddings"
  )

  # ALBERT word embeddings projector layer
  if (!is.null(self$params$embedding_size)) {
    self$word_embeddings_projector_layer <-
      keras::layer_dense(units = self$params$hidden_size)
      # custom_layer_embeddings_proj(param_list = self$params,
      #   #TODO: I think I will need to do more with names, to ensure uniqueness.
      #                              name = "word_embeddings_projector"
      # )
  }

  if (self$params$use_token_type) {
    self$token_type_embeddings_layer <- keras::layer_embedding(
      input_dim = self$params$token_type_vocab_size, # ~2L
      output_dim = position_embedding_size,
      mask_zero = FALSE,
      name = "token_type_embeddings"
    )
  }

  if (self$params$use_position_embeddings) {
    self$position_embeddings_layer <- custom_layer_position_embedding(
      param_list = self$params,
      name = "position_embeddings",
      hidden_size = position_embedding_size
    )
  }

  self$layer_norm <- custom_layer_layernorm(
    param_list = self$params,
    name = "LayerNorm"
  )
  self$dropout_layer <-
    keras::layer_dropout(rate = self$params$hidden_dropout)

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_bert_embeddings_call <- function(inputs,
                                               mask = NULL,
                                               training = NULL) {
  if (inherits(inputs, "list")) {
    if (length(inputs) != 2) {
      stop("In BertEmbeddingsLayer: ",
           "Expecting inputs to be a [input_ids, token_type_ids] list, ",
           "or else just input_ids.")
    }
    input_ids <- inputs[[1]]
    token_type_ids <- inputs[[2]]
  } else{
    input_ids <- inputs
    token_type_ids <- NULL
  }

  input_ids <- tensorflow::tf$cast(input_ids, dtype = tensorflow::tf$int32)

  embedding_output <- self$word_embeddings_layer(input_ids)

  if (!is.null(token_type_ids)) {
    token_type_ids <- tensorflow::tf$cast(token_type_ids,
                                          dtype = tensorflow::tf$int32)
    #TODO: use tf add here? No, probably not.
    embedding_output <- embedding_output +
      self$token_type_embeddings_layer(token_type_ids)
  }

  if (!is.null(self$position_embeddings_layer)) {
    seq_len <- input_ids$shape$as_list()[[2]] # check index
    eos <- embedding_output$shape$as_list()
    emb_size <- eos[[length(eos)]]

    pos_embeddings <- self$position_embeddings_layer(seq_len)

    # broadcast over all dimension but the last two [..., seq_len, width]
    broadcast_shape <- as.list(
      c(rep(1L, embedding_output$shape$ndims - 2),
        seq_len, emb_size)
    )

    embedding_output <- embedding_output +
      tensorflow::tf$reshape(pos_embeddings, broadcast_shape)
  }

  embedding_output <- self$layer_norm(embedding_output)
  embedding_output <- self$dropout_layer(embedding_output,
                                         training = training)
  # embedding_output <- keras::layer_dropout(embedding_output,
  #                                          rate = self$params$hidden_dropout)

  # ALBERT: project embeddings
  if (!is.null(self$word_embeddings_projector_layer)) {
    embedding_output <-
      self$word_embeddings_projector_layer(embedding_output)
  }

  return(embedding_output) # [B, seq_len, hidden_size]
}

#' @keywords internal
.make_custom_layer_bert_embeddings <- function() {
  layer_function <- keras::Layer(
    classname = "BertEmbeddings",

    initialize = .custom_layer_bert_embeddings_init,
    build = .custom_layer_bert_embeddings_build,
    call = .custom_layer_bert_embeddings_call,
    compute_mask <- function() {} # add this here!
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Bert Embeddings
#'
#' Create layer...
#'
#' @inheritParams custom_layer_layernorm
#' @param param_list A named list of parameter values used in defining the
#'   layer.
#'   \describe{
#'   \item{`dtype`}{The data type of the layer output. Defaults to "float32".
#'   Valid values from `tensorflow::tf$float32$name`, etc. }
#'  }
#'
#' @export
#' @md
custom_layer_bert_embeddings <- function(object,
                name = NULL,
                trainable = NULL,
                param_list = list(),
                ...) {
  keras::create_layer(layer_class = .custom_layers$bert_embeddings, # define in .onLoad
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}


