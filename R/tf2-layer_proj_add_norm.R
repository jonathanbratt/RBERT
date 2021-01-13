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


# custom layer: project, add, normalize ----------------------------------------

#' @keywords internal
.custom_layer_proj_add_norm_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))

  self$supports_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_proj_add_norm_build <- function(input_shape) {
  if (!(inherits(input_shape, "list") & length(input_shape) == 2)) {
    stop("In ProjAddNorm$build: input_shape must be length-2 list.")
    #TODO: also make sure dimensions are compatible with hidden_size.
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
#' @inheritParams custom_layer_BERT
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





