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

# SingleTransformerEncoderLayer -----------------------------------------------

#' @export
SingleTransformerEncoderLayer <- R6::R6Class(
  "SingleTransformerEncoderLayer",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  public = list(
    initialize = function(param_list = list(), ...) {
      self$params <- list(
        intermediate_size  = NULL,
        intermediate_activation = "gelu",
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
        stop("In SingleTransformerEncoderLayer$initialize: ",
             "hidden_size should be multiple of num_heads.")
      }

      self$size_per_head <- self$params$hidden_size / self$params$num_heads

      if (!is.null(self$params$size_per_head)) {
        if (self$params$size_per_head != self$size_per_head) {
          stop("In SingleTransformerEncoderLayer$initialize: ",
               "calculated size_per_head doesn't match passed value.")
        }
      }

      self$self_attention_layer <- NULL
      self$intermediate_layer <- NULL
      self$output_projector <- NULL

      self$supports_masking <- TRUE
    },

    build = function(input_shape) {
      # (input_spec?)
      self$self_attention_layer <- custom_layer_transformer_self_attention(
        param_list = self$params,
        name = "attention"
      )

      # ok, now I'm not sure again... why can't we use keras::layer_dense here? I think we can.
      self$intermediate_layer <- keras::layer_dense(
        units = self$params$intermediate_size,
        # activation = self$params$intermediate_activation,
        # activation should be string, not actual function at this point?
        activation = get_activation(self$params$intermediate_activation),
        kernel_initializer = keras::initializer_truncated_normal(
          stddev = self$params$initializer_range
        ),
        name = "intermediate"
      )

      self$output_projector <- custom_layer_projection(
        param_list = self$params,
        name = "output")

      super$build(input_shape)
    },

    call = function(inputs, mask = NULL, training = NULL) {
      layer_input <- inputs

      attention_output <- self$self_attention_layer(layer_input,
                                                    mask = mask,
                                                    training = training)
      intermediate_output <- self$intermediate_layer(attention_output)
      # in kpe/bert-for-tf2 (which I referred to heavily), the `training`
      # parameter is not included below. However, if I understand correctly,
      # it *should* be there. -JDB
      # https://github.com/kpe/bert-for-tf2/issues/18
      layer_output <- self$output_projector(list(intermediate_output,
                                                 attention_output),
                                            training = training,
                                            mask = mask)
      return(layer_output)
    }
  )
)



# wrapper function
#' @export
custom_layer_single_tranformer_encoder <- function(object,
                                                   trainable = TRUE,
                                                   param_list = list(),
                                                   ...) {
  keras::create_layer(layer_class = SingleTransformerEncoderLayer,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}

# TransformerEncoderLayer -----------------------------------------------


#' @export
TransformerEncoderLayer <- R6::R6Class(
  "TransformerEncoderLayer",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  public = list(
    initialize = function(param_list = list(), ...) {
      # lazy way of mimicking the params-flow stuff.
      # ALSO! don't forget to look up the chain to see what inherited parameters should be present

      self$params <- list(
        num_layers = NULL,
        out_layer_ndxs = NULL,   # [-1] # which layers to return. 1-indexed.
        shared_layer = FALSE,  # False for BERT, True for ALBERT
        intermediate_size  = NULL,
        intermediate_activation = "gelu",
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

      self$encoder_layers <- list()
      self$shared_layer <- NULL  # for ALBERT
      self$supports_masking <- TRUE
    },

    build = function(input_shape) {
      # (input_spec?)
      # create transformer encoder sub-layers
      if (self$params$shared_layer) {
        # ALBERT: share parameters
        self$shared_layer <- custom_layer_single_tranformer_encoder(
          param_list = self$params,
          name = "layer_shared"
        )
      } else {
        # BERT
        # NB: this will start with 1.
        layer_indices <- seq_len(self$params$num_layers)
        for (layer_index in layer_indices) {
          # Names must be 0-indexed for compatibility with existing checkpoints.
          layer_name <- paste0("layer_", layer_index - 1)
          encoder_layer <- custom_layer_single_tranformer_encoder(
            param_list = self$params,
            name = layer_name
          )
          self$encoder_layers[[layer_index]] <- encoder_layer
        }
      }

      super$build(input_shape)
    },
    call = function(inputs, mask = NULL, training = NULL) {
      layer_output <- inputs

      layer_outputs <- list()
      # NB: this will start with 1.
      layer_indices <- seq_len(self$params$num_layers)
      for (layer_index in layer_indices) {
        if (!is.null(self$encoder_layers)) {
          #BERT
          encoder_layer <- self$encoder_layers[[layer_index]]
        } else {
          #ALBERT
          encoder_layer <- self$shared_layer
        }
        layer_input <- layer_output
        layer_output <- encoder_layer(layer_input,
                                      mask = mask,
                                      training = training)
        layer_outputs[[layer_index]] <- layer_output # This will be 1-indexed.
      }

      # In pre-tf2 RBERT, returned attention matrices along with layer outputs
      # here.
      if (is.null(self$params$out_layer_ndxs)) {
        # return just the final layer
        final_output <- layer_output
      } else {
        final_output <- list()
        out_index_num <- 1
        for (index in self$params$out_layer_ndxs) {
          final_output[[out_index_num]] <- layer_outputs[index]
          out_index_num <- out_index_num + 1
        }
        # in kpe/bert-for-tf2 (python), final_output was changed to tuple here.
      }

      return(final_output)
    }
  )
)

# wrapper function
#' @export
custom_layer_tranformer_encoder <- function(object,
                                                   trainable = TRUE,
                                                   param_list = list(),
                                                   ...) {
  keras::create_layer(layer_class = TransformerEncoderLayer,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}
