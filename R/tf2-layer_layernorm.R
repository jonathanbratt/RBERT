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


# custom layer: layer normalization -------------------------------------------

#' @keywords internal
.custom_layer_layernorm_init <- function(param_list, ...) {
  self$params <- .update_list(param_list, list(...))

  self$supports_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_layernorm_build <- function(input_shape) {
  # `input_shape`: Integer vector; shape of tensor input to this layer.
  # Not sure if I need to set self$input_spec, or if that is optional.
  # I can't find the functions for creating `InputSpec` objects in
  # the R keras/tensorflow packages, nor is it done in the examples.
  # So I will just skip this part for now. -JDB
  # self$input_spec <- (something like) keras::InputSpec(shape=input_shape)


  # Get `input_shape` as proper list. Otherwise, first (NULL) dimension
  # isn't included in `length` for some reason.
  input_shape_l <- input_shape$as_list()
  self$gamma <- self$add_weight(name = "gamma",
                                shape = input_shape_l[length(input_shape_l)],
                                initializer = keras::initializer_ones(),
                                trainable = TRUE)

  self$beta <- self$add_weight(name = "beta",
                               shape = input_shape_l[length(input_shape_l)],
                               initializer = keras::initializer_zeros(),
                               trainable = TRUE)

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_layernorm_call <- function(inputs, ...) {

  #TODO double check the axes indexing conventions here.
  mv <- tensorflow::tf$nn$moments(inputs, axes = -1L, keepdims = TRUE)
  mean <- mv[[1]]
  var <- mv[[2]] # uses N, not N-1, in denominator. Just FYI.

  inv <- self$gamma*tensorflow::tf$math$rsqrt(var + self$params$epsilon)

  res <- inputs*tensorflow::tf$cast(inv, inputs$dtype) +
    tensorflow::tf$cast(self$beta - mean*inv, inputs$dtype)

  return(res)
}

#' @keywords internal
.make_custom_layer_layernorm <- function() {
  layer_function <- keras::Layer(
    classname = "LayerNorm",

    initialize = .custom_layer_layernorm_init,
    build = .custom_layer_layernorm_build,
    call = .custom_layer_layernorm_call,

    #TODO Do we need this method?
    compute_mask = function(inputs, mask = NULL) {
      return(mask)
    }
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: Layer Normalization
#'
#' Create a layer that applies layer normalization to the previous layer output.
#' Note that this layer contains trainable parameters.
#'
#' @param object Model or layer object.
#' @param name Character; An optional name for the layer. Must be unique in a
#'   model.
#' @param trainable Logical; whether the layer weights will be updated during
#'   training.
#' @param param_list A named list of parameter values used in defining the
#'   layer.
#'   \describe{
#'    \item{`epsilon`}{Numeric; small value added to denominators to avoid
#'    dividing by zero. Defaults to `1e-12`.}
#'    \item{`dtype`}{The data type of the layer output. Defaults to "float32".
#'    Valid values from `tensorflow::tf$float32$name`, etc. }
#'   }
#'
#' @source \url{https://arxiv.org/pdf/1607.06450.pdf}
#' @export
#' @md
custom_layer_layernorm <- function(object,
                                   name = NULL,
                                   trainable = NULL,
                                   param_list = list(),
                                   ...) {
  keras::create_layer(layer_class = .custom_layers$layernorm,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}

