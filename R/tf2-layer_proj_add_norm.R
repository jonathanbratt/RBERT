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


# custom layer: project, add, normalize ----------------------------------------

#' @keywords internal
.custom_layer_proj_add_norm_init <- function(param_list = list(), ...) {
  # Quick way of mimicking the params-flow structure.

  #TODO: Double check which params are used.

  self$params <- list(
    hidden_size = NULL,
    hidden_dropout = 0.1,
    initializer_range = 0.02,
    trainable = TRUE,
    name = "ProjAddNorm",
    dtype = tensorflow::tf$float32$name,
    dynamic = FALSE
  )
  self$params <- .update_list(self$params, param_list)
  self$params <- .update_list(self$params, list(...))

  self$dense <- NULL
  self$dropout <- NULL
  self$layer_norm <- NULL

  self$supports_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_proj_add_norm_build <- function(input_shape) {
  if (!(inherits(input_shape, "list") & length(input_shape) == 2)) {
    stop("In ProjAddNorm$build: input_shape must be length-2 list.")
    #TODO: also make sure dimensions are compatible with hidden_size?
  }

  self$dense <- keras::layer_dense(
    units = self$params$hidden_size,
    kernel_initializer = keras::initializer_truncated_normal(
      stddev = self$params$initializer_range
    ),
    name = "dense"
  )

  self$dropout <- keras::layer_dropout(rate = self$params$hidden_dropout)
  self$layer_norm <- custom_layer_layernorm(
    param_list = self$params,
    name = "LayerNorm"
  )

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_proj_add_norm_call <- function(inputs,
                                             mask = NULL,
                                             training = NULL) {
  output <- inputs[[1]]
  residual <- inputs[[2]]

  output <- self$dense(output)
  output <- self$dropout(output, training = training)
  output <- self$layer_norm(output + residual)

  return(output)
}

#' @keywords internal
.make_custom_layer_proj_add_norm <- function() {
  layer_function <- keras::Layer(
    classname = "ProjAddNorm",

    initialize = .custom_layer_proj_add_norm_init,
    build = .custom_layer_proj_add_norm_build,
    call = .custom_layer_proj_add_norm_call
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Project, Add, Normalize
#'
#' Create a layer that, given two input layers, applies a dense layer projection
#' (followed by dropout) to the first input, then adds the second (as a
#' residual) and normalizes the sum.
#'
#' @inheritParams custom_layer_layernorm
#' @param param_list A named list of parameter values used in defining the
#'   layer.
#'   \describe{
#'   \item{`hidden_size`}{Integer; The size of the output. Should match the
#'     size of the second input.}
#'   \item{`hidden_dropout`}{Numeric; the dropout rate (fraction to drop)
#'     applied after the dense layer projection.}
#'   \item{`initializer_range`}{Numeric; the value passed in as the `stddev`
#'     parameter to the `initializer_truncated_normal` in the dense layer
#'     initializer.}
#'   \item{`dtype`}{The data type of the layer output. Defaults to "float32".
#'   Valid values from `tensorflow::tf$float32$name`, etc. }
#'  }
#'
#' @export
#' @md
custom_layer_proj_add_norm <- function(object,
                                   name = NULL,
                                   trainable = NULL,
                                   param_list = list(),
                                   ...) {
  keras::create_layer(layer_class = .custom_layers$proj_add_norm,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}





