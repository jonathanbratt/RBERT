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

# custom layer: transformer_encoder_single ----------------------------------------

#' @keywords internal
.custom_layer_BERT_init <- function(param_list = list(),
                                    ...) {
  self$params <- list(
    vocab_size = NULL,
    use_token_type = TRUE,
    use_position_embeddings = TRUE,
    token_type_vocab_size = 2L,
    embedding_size = NULL,   # NULL for BERT, not NULL for ALBERT
    max_position_embeddings = 512L,
    num_layers = NULL,
    return_all_layers = TRUE,
    shared_layer = FALSE,  # False for BERT, True for ALBERT
    intermediate_size  = NULL,
    intermediate_activation = "gelu",
    hidden_size = 768L,
    num_heads = NULL,
    hidden_dropout = 0.1,
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

  self$embeddings_layer <- NULL
  self$encoders_layer <- NULL

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
  if (is.null(mask)) {
    mask <- self$embeddings_layer$compute_mask(inputs)
  }

  embedding_output <- self$embeddings_layer(inputs,
                                            mask = mask,
                                            training = training)
  output <- self$encoders_layer(embedding_output,
                                mask = mask,
                                training = training)
  return(output)   # [B, seq_len, hidden_size]
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
#'   layer.
#'   \describe{
#'   \item{`dtype`}{The data type of the layer output.
#'     Defaults to "float32". Valid values from `tensorflow::tf$float32$name`,
#'     etc. }
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





