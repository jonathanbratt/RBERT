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
ProjectionLayer <- R6::R6Class(
  "ProjectionLayer",
  inherit = keras::KerasLayer,
  lock_objects = FALSE,

  public = list(
    initialize = function(param_list = list(), ...) {
      # Quick way of mimicking the params-flow structure.
      # Some params may not be used? Check.

      self$params <- list(
        hidden_size = NULL,
        hidden_dropout = 0.1,
        initializer_range = 0.02,
        trainable = TRUE,
        name = NULL,
        dtype = tensorflow::tf$float32$name,
        dynamic = FALSE
      )
      self$params <- .update_list(self$params, param_list)
      self$params <- .update_list(self$params, list(...))

      self$dense <- NULL
      self$dropout <- NULL
      self$layer_norm <- NULL

      self$supports_masking <- TRUE
    },

    build = function(input_shape) {
      # Not sure if I need to set self$input_spec, or if that is optional.
      if (!(inherits(input_shape, "list") & length(input_shape) == 2)) {
        stop("In ProjectionLayer$build: input_shape must be length-2 list.")
      }

      # self$dense <- keras:::keras$layers$Dense(
      self$dense <- keras::layer_dense(
      units = self$params$hidden_size,
        kernel_initializer = keras::initializer_truncated_normal(
          stddev = self$params$initializer_range
        ),
        name = "dense"
      )

      self$dropout <- keras::layer_dropout(rate = self$params$hidden_dropout) # this belongs in call, not build.
      self$layer_norm <- custom_layer_normalization(
        param_list = self$params,
        name = "LayerNorm" # check to make sure the function handles this argument correctly?
      )

      super$build(input_shape)
    },

    call = function(inputs, mask = NULL, training = NULL) {
      output <- inputs[[1]]
      residual <- inputs[[2]]

      output <- self$dense(output)
      output <- self$dropout(output, training = training)
      # output <- keras::layer_dropout(output, rate = self$params$hidden_dropout)
      output <- self$layer_norm(output + residual)
      return(output)
    }
  )
)


# wrapper function
#' @export
custom_layer_projection <- function(object,
                             trainable = TRUE,
                             param_list = list(),
                             ...) {
  keras::create_layer(layer_class = ProjectionLayer,
                      object = object,
                      args = list(
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      ))
}


