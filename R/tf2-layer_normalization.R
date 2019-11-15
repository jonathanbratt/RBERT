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

#' @export
LayerNormalization <- R6::R6Class(
  "LayerNormalization",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  # Layer normalization layer from arXiv:1607.06450.
  # See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
  # See: https://github.com/CyberZHG/keras-layer-normalization
  # See: tf.contrib.layers.layer_norm

  public = list(
    initialize = function(param_list = list(), ...) {
      # For now, explicitly support *only* TF2.
      if (!.tf2()) {
        stop("LayerNormalization requires TensorFlow 2 to be installed.")
      }
      # Quick way of mimicking the params-flow structure.
      # Some params may not be used? Check.

      self$params <- list(
        epsilon = 1e-12,
        initializer_range = 0.02,
        trainable = TRUE,
        name = "LayerNorm",
        dtype = tensorflow::tf$float32$name,
        dynamic = FALSE
      )
      # The initialization method takes a param_list argument, as well as any
      # other named arguments. These parameters are used to update the `params`
      # property.
      self$params <- .update_list(self$params, param_list)
      self$params <- .update_list(self$params, list(...))

      self$gamma  <- NULL
      self$beta  <- NULL
      self$supports_masking <- TRUE
    },

    build = function(input_shape) {
      # `input_shape`: Integer vector; shape of tensor input to this layer.
      # Not sure if I need to set self$input_spec, or if that is optional.
      # I can't find the functions for creating `InputSpec` objects in
      # the R keras/tensorflow packages, nor is it done in the examples.
      # So I will just skip this part for now. -JDB
      # self$input_spec <- (something like) keras::InputSpec(shape=input_shape)

      self$gamma <- self$add_weight(name = "gamma",
                                    shape = input_shape[[length(input_shape)]],
                                    initializer = keras::initializer_ones(),
                                    trainable = TRUE)

      self$beta <- self$add_weight(name = "beta",
                                   shape = input_shape[[length(input_shape)]],
                                   initializer = keras::initializer_zeros(),
                                   trainable = TRUE)

      super$build(input_shape)
    },
    call = function(inputs, mask = NULL) {
      # double check the axes indexing conventions here.
      mv <- tensorflow::tf$nn$moments(inputs, axes = -1L, keepdims = TRUE)
      mean <- mv[[1]]
      var <- mv[[2]] # uses N, not N-1, in denominator. Just FYI.

      inv <- self$gamma*tensorflow::tf$math$rsqrt(var + self$params$epsilon)

      res <- inputs*tensorflow::tf$cast(inv, inputs$dtype) +
        tensorflow::tf$cast(self$beta - mean*inv, inputs$dtype)

      return(res)
    },
    compute_output_shape = function(input_shape) {
      return(input_shape)
    },
    compute_mask = function(inputs, mask = NULL) {
      return(mask)
    }
  )
)


# wrapper function
#' @export
custom_layer_normalization <- function(object,
                             trainable = TRUE,
                             param_list = list(),
                             ...) {
  keras::create_layer(layer_class = LayerNormalization,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}
