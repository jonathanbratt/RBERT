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

# custom layer: BERT ----------------------------------------

#' @keywords internal
.custom_layer_BERT_init <- function(param_list, ...) {
  # Individually named parameters take precedence over values in param_list.
  self$params <- .update_list(param_list, list(...))

  self$support_masking <- TRUE

  super()$`__init__`(name = self$params$name)
}

#' @keywords internal
.custom_layer_BERT_build <- function(input_shape) {
  self$embeddings_layer <-  custom_layer_bert_embeddings(
    param_list = self$params,
    name = "embeddings"
  )

  self$encoders_layer <- custom_layer_transformer_encoder(
    param_list = self$params,
    name = "encoder"
  )

  super()$build(input_shape)
}


#' @keywords internal
.custom_layer_BERT_call <- function(inputs,
                                    mask = NULL,
                                    training = NULL) {
  embedding_output <- self$embeddings_layer(inputs,
                                            mask = mask,
                                            training = training)

  # Compute the mask here instead of using compute_mask method.
  input_ids <- inputs[[1]] # inputs is a [input_ids, token_type_ids] list.
  mask <- tensorflow::tf$not_equal(input_ids, 0L)

  output <- self$encoders_layer(embedding_output,
                                mask = mask,
                                # mask = mask2,
                                training = training)
  # The order of elements in a combined list can be unpredictable here,
  # so set it explicitly.
  return(list("initial_embeddings" = embedding_output,
              "layer_output" = output$output,
              "attention_matrix" = output$attention))
}

#' @keywords internal
.make_custom_layer_BERT <- function() {
  layer_function <- keras::Layer(
    classname = "BERT",

    initialize = .custom_layer_BERT_init,
    build = .custom_layer_BERT_build,
    call = .custom_layer_BERT_call
  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' Custom Layer: BERT model
#'
#' Create a BERT (or ALBERT) model.
#'
#'
#' @inheritParams custom_layer_layernorm
#' @param param_list A named list of parameter values used in defining the
#'   layer. *standardize this here*
#'   \describe{
#'   \item{`xxx`}{description }
#'   }
#'
#' @export
#' @md
custom_layer_BERT <- function(object,
                              name = NULL,
                              trainable = NULL,
                              param_list = list(),
                              ...) {
  keras::create_layer(layer_class = .custom_layers$BERT,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
  )
}





