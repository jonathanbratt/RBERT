# Copyright 2021 Bedford Freeman & Worth Pub Grp LLC DBA Macmillan Learning.
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

# custom layer: transformer_encoder_single ----------------------------------------

#' @keywords internal
.custom_layer_transformer_encoder_single_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))

  if (self$params$hidden_size %% self$params$num_heads != 0) {
    stop("In .custom_layer_transformer_encoder_single_init: ",
         "hidden_size should be multiple of num_heads.")
  }

  size_per_head <- self$params$hidden_size / self$params$num_heads
  if (self$params$size_per_head != size_per_head) {
    stop("In .custom_layer_transformer_encoder_single_init: ",
         "calculated size_per_head doesn't match passed value.")
  }

  self$supports_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_transformer_encoder_single_build <- function(input_shape) {
  self$self_attention_layer <- custom_layer_attention(
    param_list = self$params,
    name = "attention"
  )

  self$intermediate_layer <- keras::layer_dense(
    units = self$params$intermediate_size,
    # Custom activation needs to be passed as actual function, not string.
    activation = get_activation(self$params$intermediate_activation),
    kernel_initializer = keras::initializer_truncated_normal(
      stddev = self$params$initializer_range
    ),
    # So variable names match canonical. Slightly questionable. :)
    name = "intermediate/dense"
  )

  self$output_projector <- custom_layer_proj_add_norm(
    param_list = self$params,
    name = "output"
  )

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_transformer_encoder_single_call <- function(inputs,
                                                          mask = NULL,
                                                          training = NULL) {
  # The attention layer also outputs the attention prob matrix.
  attention_output_and_probs <- self$self_attention_layer(inputs,
                                                          mask = mask,
                                                          training = training)
  attention_output <- attention_output_and_probs[[1]]
  attention_probs <- attention_output_and_probs[[2]]
  intermediate_output <- self$intermediate_layer(attention_output)

  layer_output <- self$output_projector(list(intermediate_output,
                                             attention_output),
                                        mask = mask)
  return(list(layer_output, attention_probs))
}

#' @keywords internal
.make_custom_layer_transformer_encoder_single <- function() {
  layer_function <- keras::Layer(
    classname = "Transformer",

    initialize = .custom_layer_transformer_encoder_single_init,
    build = .custom_layer_transformer_encoder_single_build,
    call = .custom_layer_transformer_encoder_single_call
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Single Transformer Encoder Layer
#'
#' Create single layer of a transformer-based encoder.
#'
#' @inheritParams custom_layer_BERT
#'
#' @export
#' @md
custom_layer_transformer_encoder_single <- function(object,
                                                    name = NULL,
                                                    trainable = NULL,
                                                    param_list = list(),
                                                    ...) {
  keras::create_layer(layer_class = .custom_layers$transformer_encoder_single,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}


# custom layer: transformer_encoder ----------------------------------------

#' @keywords internal
.custom_layer_transformer_encoder_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))

  # self$shared_layer <- NULL  # for ALBERT
  self$supports_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_transformer_encoder_build <- function(input_shape) {
  # create transformer encoder sub-layers
  encoder_layers <- NULL
  if (self$params$shared_layer) {
    # ALBERT: share parameters
    self$shared_layer <- custom_layer_transformer_encoder_single(
      param_list = self$params,
      name = "layer_shared"
    )
  } else {
    # BERT
    # NB: this will start with 1.
    layer_indices <- seq_len(self$params$num_layers)

    encoder_layers <- vector("list", self$params$num_layers)
    for (layer_index in layer_indices) {
      # Names must be 0-indexed for compatibility with existing checkpoints.
      layer_name <- paste0("layer_", layer_index - 1)
      encoder_layer <- custom_layer_transformer_encoder_single(
        param_list = self$params,
        name = layer_name
      )
      encoder_layers[[layer_index]] <- encoder_layer
    }
  }

  # Lists as class variables are handled a bit strangely by python... explain?
  self$encoder_layers <- encoder_layers

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_transformer_encoder_call <- function(inputs,
                                                   mask = NULL,
                                                   training = NULL) {
  layer_output <- inputs
  layer_output_all <- list()
  attention_probs_all <- list()
  # NB: this will start with 1.
  layer_indices <- seq_len(self$params$num_layers)
  for (layer_index in layer_indices) {
    if (!self$params$shared_layer) {
      #BERT
      # The "list" self$encoder_layers has been converted into a python object
      # by this point (a tensorflow ListWrapper, to be specific), and is now
      # zero-indexed, hence the `- 1`.
      encoder_layer <- self$encoder_layers[[layer_index - 1]]
    } else {
      #ALBERT
      encoder_layer <- self$shared_layer
    }
    layer_input <- layer_output
    layer_output_and_probs <- encoder_layer(layer_input,
                                            mask = mask,
                                            training = training)
    layer_output <- layer_output_and_probs[[1]]
    attention_probs <- layer_output_and_probs[[2]]
    # These will be 1-indexed.
    layer_output_all[[layer_index]] <- layer_output
    attention_probs_all[[layer_index]] <- attention_probs
  }

  # I think it's probably best to just always return all the layers. -JDB
  if (self$params$return_all_layers) {
    final_output <- list("output" = layer_output_all,
                         "attention" = attention_probs_all)
  } else {
    # return just the final layer
    final_output <- layer_output_and_probs
  }

  return(final_output)
}

#' @keywords internal
.make_custom_layer_transformer_encoder <- function() {
  layer_function <- keras::Layer(
    classname = "Transformer",

    initialize = .custom_layer_transformer_encoder_init,
    build = .custom_layer_transformer_encoder_build,
    call = .custom_layer_transformer_encoder_call
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Transformer Encoder Layer
#'
#' Create a multi-layer transformer-based encoder.
#'
#'
#' @inheritParams custom_layer_BERT
#'
#' @export
#' @md
custom_layer_transformer_encoder <- function(object,
                                             name = NULL,
                                             trainable = NULL,
                                             param_list = list(),
                                             ...) {
  keras::create_layer(layer_class = .custom_layers$transformer_encoder,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}




