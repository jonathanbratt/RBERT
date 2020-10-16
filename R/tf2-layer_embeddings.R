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
.custom_layer_position_embedding_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))

  self$supports_masking <- TRUE
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
.custom_layer_bert_embeddings_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))
  if (isFALSE(self$params$use_token_type)) {
    warning("use_token_type FALSE is not supported; treating as TRUE.")
  }
  if (isFALSE(self$params$use_position_embeddings)) {
    warning("use_position_embeddings FALSE is not supported; treating as TRUE.")
  }

  self$word_embeddings_projector_layer <- NULL  # for ALBERT

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
  }

  self$token_type_embeddings_layer <- keras::layer_embedding(
    input_dim = self$params$token_type_vocab_size, # ~2L
    output_dim = position_embedding_size,
    mask_zero = FALSE,
    name = "token_type_embeddings"
  )

  self$position_embeddings_layer <- custom_layer_position_embedding(
    param_list = self$params,
    name = "position_embeddings",
    hidden_size = position_embedding_size
  )

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
      stop("In bert_embeddings layer: ",
           "Input is not a length-2 list.",
           "Expecting inputs to be a [input_ids, token_type_ids] list.")
    }
    input_ids <- inputs[[1]]
    token_type_ids <- inputs[[2]]
  } else{
    stop("In bert_embeddings layer: ",
         "Input is not a list.",
         "Expecting inputs to be a [input_ids, token_type_ids] list.")
  }

  # base token embeddings...
  input_ids <- tensorflow::tf$cast(input_ids, dtype = tensorflow::tf$int32)
  embedding_output <- self$word_embeddings_layer(input_ids)

  # token type embeddings...
  token_type_ids <- tensorflow::tf$cast(token_type_ids,
                                        dtype = tensorflow::tf$int32)
  embedding_output <- embedding_output +
    self$token_type_embeddings_layer(token_type_ids)

  # token postion embeddings...
  # Position indices are already implicit in the tensor structure, so we
  # don't need to pass in a tensor of ids, just the sequence length.
  seq_len <- input_ids$shape$as_list()[[2]] # check index
  eos <- embedding_output$shape$as_list()
  emb_size <- eos[[length(eos)]]

  pos_embeddings <- self$position_embeddings_layer(seq_len)

  # Broadcast over all dimensions but the last two [..., seq_len, width].
  # Because we construct this embedding layer from scratch, we do this to
  # bring it to the right shape.
  broadcast_shape <- as.list(
    c(rep(1L, embedding_output$shape$ndims - 2),
      seq_len, emb_size)
  )

  embedding_output <- embedding_output +
    tensorflow::tf$reshape(pos_embeddings, broadcast_shape)

  embedding_output <- self$layer_norm(embedding_output)
  embedding_output <- self$dropout_layer(embedding_output,
                                         training = training)

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
    call = .custom_layer_bert_embeddings_call
    # I can't figure out how to invoke the compute_mask method so that it works.
    # Move the computation of the mask into the bert layer `call` method. -JDB
    # compute_mask <- function(inputs, mask = NULL) {}
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
  keras::create_layer(layer_class = .custom_layers$bert_embeddings,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}


