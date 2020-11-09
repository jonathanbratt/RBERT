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


# custom layer: attention ----------------------------------------

#' @keywords internal
.custom_layer_attention_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))

  self$supports_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_attention_build <- function(input_shape) {
  # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head

  ki <- keras::initializer_truncated_normal(
    stddev = self$params$initializer_range
  )

  self$query_layer <- keras::layer_dense(
    units = self$params$hidden_size,
    activation = self$params$query_activation,
    kernel_initializer = ki,
    # So variable names match canonical. Slightly questionable. :)
    name = "self/query"
  )

  self$key_layer <- keras::layer_dense(
    units = self$params$hidden_size,
    activation = self$params$key_activation,
    kernel_initializer = ki,
    name = "self/key"
  )

  self$value_layer <- keras::layer_dense(
    units = self$params$hidden_size,
    activation = self$params$value_activation,
    kernel_initializer = ki,
    name = "self/value"
  )

  self$dropout_layer <- keras::layer_dropout(
    rate = self$params$attention_dropout)

  self$attention_projector <- custom_layer_proj_add_norm(
    object = NULL,
    param_list = self$params,
    name = "output"
  )

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_attention_call <- function(inputs,
                                         mask = NULL,
                                         training = NULL) {
  from_tensor <- inputs
  to_tensor <- inputs

  if (is.null(mask)) {
    sh <- get_shape_list(from_tensor)
    mask <- tensorflow::tf$ones(sh[1:2], dtype = tensorflow::tf$int32)
  }

  attention_mask <- .create_attention_mask(
    tensorflow::tf$shape(input = from_tensor),
    mask
  )

  #  from_tensor shape is: [batch_size, from_seq_length, from_width]
  input_shape <- get_shape_list(from_tensor)

  batch_size <- input_shape[[1]]
  from_seq_len <- input_shape[[2]]
  from_width <- input_shape[[3]]  # hidden_size
  to_seq_len <- from_seq_len

  query <- self$query_layer(from_tensor)  # [B,F, N*H]
  key <- self$key_layer(to_tensor)        # [B,T, N*H]
  value <- self$value_layer(to_tensor)    # [B,T, N*H]

  # [B, F, N*H] -> [B, N, F, H]
  query <- .transpose_for_scores(
    input_tensor = query,
    batch_size = batch_size,
    num_attention_heads = self$params$num_heads,
    seq_length = from_seq_len,
    width = self$params$size_per_head
  ) # now: [B, N, F, H]

  key <- .transpose_for_scores(
    input_tensor = key,
    batch_size = batch_size,
    num_attention_heads = self$params$num_heads,
    seq_length = to_seq_len,
    width = self$params$size_per_head
  ) # [B, N, T, H]

  value <- .transpose_for_scores(
    input_tensor = value,
    batch_size = batch_size,
    num_attention_heads = self$params$num_heads,
    seq_length = to_seq_len,
    width = self$params$size_per_head
  ) # [B, N, T, H]

  attention_scores <- tensorflow::tf$matmul(query, key, transpose_b = TRUE)
  # scores dimensions: [B, N, F, T]

  attention_scores <- tensorflow::tf$multiply(
    attention_scores,
    tensorflow::tf$math$rsqrt(as.numeric(self$params$size_per_head))
  )

  if (!is.null(attention_mask)) {
    # The axis argument is zero-indexed, so the expanded dimension is the
    # *second*, not the first: [B, 1, F, T]
    attention_mask <- tensorflow::tf$expand_dims(attention_mask,
                                                 axis = list(1L))
    # Since attention_mask is 1.0 for positions we want to attend and 0.0
    # for masked positions, this operation will create a tensor which is 0.0
    # for positions we want to attend and -10000.0 for masked positions.
    adder <- self$params$negative_infinity *
      (1.0 - tensorflow::tf$cast(attention_mask, tensorflow::tf$float32))
    # ...since we are adding it to the raw scores before the softmax, this
    # is effectively the same as removing these entirely.
    attention_scores <- attention_scores + adder
  }

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs <-  tensorflow::tf$nn$softmax(attention_scores)

  # "This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper."
  attention_probs <- self$dropout_layer(attention_probs) # [B, N, F, T]

  # `context_layer` = [B, N, F, H]
  #TODO: come up with better name than "context_layer"?
  context_layer <- tensorflow::tf$matmul(attention_probs, value)

  # `context_layer` = [B, F, N, H]
  context_layer <- tensorflow::tf$transpose(context_layer,
                                            perm = list(0L, 2L, 1L, 3L))

  # [B, F, N*H]
  output_shape <- list(
    batch_size,
    from_seq_len,
    self$params$hidden_size # N*H is just hidden_size
  )
  context_layer <- tensorflow::tf$reshape(context_layer, output_shape)

  # Add residual and layer norm here, rather than making whole other layer.
  # I think the mask parameter needs to be explicitly passed, because
  # it may have been changed in this layer.
  attention_output <- self$attention_projector(list(context_layer,
                                                    inputs),
                                               mask = mask)

  return(list(attention_output, attention_probs))
}

#' @keywords internal
.make_custom_layer_attention <- function() {
  layer_function <- keras::Layer(
    classname = "Attention",

    initialize = .custom_layer_attention_init,
    build = .custom_layer_attention_build,
    call = .custom_layer_attention_call
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Attention
#'
#' Create first part of self attention layer. Takes as input an embeddings layer
#' (could be the output of a previous attention layer), performs self attention,
#' adds input layer back via residual connection, and applies layer
#' normalization.
#'
#' In an encoder, this layer is typically followed by an intermediate dense
#' layer, with a redsidual connection before another layer normalization.
#'
#' Note that this layer implements more of the attention mechanism than
#' does `keras::layer_attention`.
#'
#' @inheritParams custom_layer_BERT
#'
#' @export
#' @md
custom_layer_attention <- function(object,
                                   name = NULL,
                                   trainable = NULL,
                                   param_list = list(),
                                   ...) {
  keras::create_layer(layer_class = .custom_layers$attention,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}
