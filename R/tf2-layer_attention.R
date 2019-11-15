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

# AttentionLayer ---------------------------------------------------------------

#' @export
AttentionLayer <- R6::R6Class(
  "AttentionLayer",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  public = list(
    initialize = function(param_list = list(), ...) {
      self$params <- list(
        num_heads = NULL,
        size_per_head = NULL,
        initializer_range = 0.02,
        query_activation = NULL,
        key_activation = NULL,
        value_activation = NULL,
        attention_dropout = 0.1,
        negative_infinity = -10000.0,
        trainable = TRUE,
        name = NULL,
        dtype = tensorflow::tf$float32$name,
        dynamic = FALSE
      )
      self$params <- .update_list(self$params, param_list)
      self$params <- .update_list(self$params, list(...))

      self$query_activation <- self$params$query_activation
      self$key_activation <- self$params$key_activation
      self$value_activation <- self$params$value_activation

      self$query_layer <- NULL
      self$key_layer <- NULL
      self$value_layer <- NULL

      self$dropout_layer <- NULL

      self$supports_masking <- TRUE
    },

    build = function(input_shape) {
      # (input_spec?)
      # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
      dense_units <- self$params$num_heads * self$params$size_per_head # N*H
      ir <- self$params$initializer_range

      self$query_layer <- keras::layer_dense(
        units = dense_units,
        activation = self$query_activation,
        kernel_initializer = keras::initializer_truncated_normal(stddev = ir),
        name = "query"
      )

      self$key_layer <- keras::layer_dense(
        units = dense_units,
        activation = self$key_activation,
        kernel_initializer = keras::initializer_truncated_normal(stddev = ir),
        name = "key"
      )

      self$value_layer <- keras::layer_dense(
        units = dense_units,
        activation = self$value_activation,
        kernel_initializer = keras::initializer_truncated_normal(stddev = ir),
        name = "value"
      )

      self$dropout_layer <- keras::layer_dropout(
        rate = self$params$attention_dropout)

      super$build(input_shape)
    },

    call = function(inputs, mask = NULL, training = NULL) {
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
      # input_shape <- tensorflow::tf$shape(input = from_tensor)
      input_shape <- get_shape_list(from_tensor)
      batch_size <- input_shape[[1]]
      from_seq_len <- input_shape[[2]]
      from_width <- input_shape[[3]]
      to_seq_len <- from_seq_len

      query <- self$query_layer(from_tensor)  # [B,F, N*H]
      key <- self$key_layer(to_tensor)        # [B,T, N*H]
      value <- self$value_layer(to_tensor)    # [B,T, N*H]

      # Confirm use of existing transpose_for_scores here.
      # [B, F, N*H] -> [B, N, F, H]
      query <- transpose_for_scores(
        input_tensor = query,
        batch_size = batch_size,
        num_attention_heads = self$params$num_heads,
        seq_length = from_seq_len,
        width = self$params$size_per_head
      ) # now: [B, N, F, H]

      key <- transpose_for_scores(
        input_tensor = key,
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

      # This is actually dropping out entire tokens to attend to, which might
      # # seem a bit unusual, but is taken from the original Transformer paper.
      attention_probs <- self$dropout_layer(attention_probs) # [B, N, F, T]

      value <- RBERT:::transpose_for_scores(
        input_tensor = value,
        batch_size = batch_size,
        num_attention_heads = self$params$num_heads,
        seq_length = to_seq_len,
        width = self$params$size_per_head
      ) # [B, N, T, H]

      # `context_layer` = [B, N, F, H]
      context_layer <- tensorflow::tf$matmul(attention_probs, value)

      # `context_layer` = [B, F, N, H]
      context_layer <- tensorflow::tf$transpose(context_layer,
                                                perm = list(0L, 2L, 1L, 3L))

      # At this point in original RBERT, there was a switch for
      # returning a 2D tensor (or 4D, which was the default).
      # Now, a 3D tensor is standard. Maybe change back to 4D?
      # Also, this is where we have the attention probabilities,
      # and modified the code to include them in the return.

      # [B, F, N*H]
      output_shape <- list(batch_size,
                           from_seq_len,
                           as.integer(self$params$num_heads *
                                        self$params$size_per_head))
      context_layer <- tensorflow::tf$reshape(context_layer, output_shape)
      return(context_layer) # edit to include attention somewhere?
    },

    compute_output_shape = function(self, input_shape) {
      from_shape <- input_shape
      output_shape <- list(from_shape[[1]],
                           from_shape[[2]],
                           as.integer(self$params$num_heads *
                                        self$params$size_per_head))
      return(output_shape) # [B, F, N*H]
    },

    compute_mask = function(self, inputs, mask = NULL) {
      return(mask) # [B, F]
    }
  )
)

# wrapper function
#' @export
custom_layer_attention <- function(object,
                                    trainable = TRUE,
                                    param_list = list(),
                                    ...) {
  keras::create_layer(layer_class = AttentionLayer,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}



# TransformerSelfAttentionLayer -----------------------------------------------

#' @export
TransformerSelfAttentionLayer <- R6::R6Class(
  "TransformerSelfAttentionLayer",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  public = list(
    initialize = function(param_list = list(), ...) {
      self$params <- list(
        hidden_size = NULL,
        num_heads = NULL,
        hidden_dropout = NULL,
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

      if (self$params$hidden_size %% self$params$num_heads != 0) {
        stop("In TransformerSelfAttentionLayer$initialize: ",
             "hidden_size ", self$params$hidden_size,
             "is not a multiple of num_heads ", self$params$num_heads, ".")
      }
      self$size_per_head <- self$params$hidden_size / self$params$num_heads

      if (!is.null(self$params$size_per_head)) {
        if (self$params$size_per_head != self$size_per_head) {
          stop("In TransformerSelfAttentionLayer$initialize: ",
               "calculated size_per_head doesn't match passed value.")
        }
      }

      self$attention_layer <- NULL
      self$attention_projector <- NULL
      self$supports_masking <- TRUE
    },

    build = function(input_shape) {
      # (input_spec?)
      self$attention_layer <- custom_layer_attention(
        object = NULL,
        param_list = self$params,
        size_per_head = self$size_per_head,
        name = "self"
      )
      self$attention_projector <- custom_layer_projection(
        object = NULL,
        param_list = self$params,
        name = "output"
      )
      super$build(input_shape)
    },

    call = function(inputs, mask = NULL, training = NULL) {
      layer_input <- inputs
      attention_head <- self$attention_layer(layer_input,
                                             mask = mask,
                                             training = training)
      attention_output <- self$attention_projector(list(attention_head,
                                                        layer_input),
                                                   mask = mask,
                                                   training = training)
      return(attention_output)
    }
  )
)

# wrapper function
#' @export
custom_layer_transformer_self_attention <- function(object,
                                    trainable = TRUE,
                                    param_list = list(),
                                    ...) {
  keras::create_layer(layer_class = TransformerSelfAttentionLayer,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}

