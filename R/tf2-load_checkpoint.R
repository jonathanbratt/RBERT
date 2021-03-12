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

# These functions are very much in draft form.

# get_params_from_checkpoint ---------------------------------------------


#' Get Model Parameters from Checkpoint
#'
#' Read the parameters for a BERT model from the specified checkpoint.
#'
#' @param ckpt_dir Character; path to checkpoint directory. If specified, any
#'   other checkpoint files required by this function (\code{vocab_file},
#'   \code{bert_config_file}, or \code{init_checkpoint}) will default to
#'   standard filenames within \code{ckpt_dir}.
#' @param bert_config_file Character; the path to a json config file.
#'
#' @return A list of BERT model parameters.
#'
#' @export
get_params_from_checkpoint <- function(
  ckpt_dir = NULL,
  bert_config_file = find_config(ckpt_dir)
) {
  bc <- jsonlite::fromJSON(bert_config_file)

  size_per_head <- bc$hidden_size / bc$num_attention_heads
  if (as.integer(size_per_head) != size_per_head) {
    stop("Inconsistent size per head implied.")
  }
  # some parameters need to be renamed...
  #TODO: rename what we can to be more consistent with original BERT
  #conventions, then streamline this maybe?
  bert_params <- list(
    num_layers = bc$num_hidden_layers,
    num_heads = bc$num_attention_heads,
    hidden_size = bc$hidden_size,
    hidden_dropout = bc$hidden_dropout_prob,
    attention_dropout = bc$attention_probs_dropout_prob,

    intermediate_size = bc$intermediate_size,
    intermediate_activation = bc$hidden_act,

    vocab_size = bc$vocab_size,
    token_type_vocab_size = bc$type_vocab_size,
    max_position_embeddings = bc$max_position_embeddings,

    embedding_size = bc$embedding_size,
    shared_layer = !is.null(bc$embedding_size),
    # below are other params, not from config
    initializer_range = 0.02,
    size_per_head = size_per_head,
    query_activation = NULL,
    key_activation = NULL,
    value_activation = NULL,
    negative_infinity = -10000.0,
    epsilon = 1e-12,
    trainable = TRUE,
    dynamic = FALSE,
    name = "bert",
    return_all_layers = TRUE
  )
  return(bert_params)
}



# load_checkpoint_weights --------------------------------------------
#

#' Load Weights from Checkpoint
#'
#' Updates the given BERT model (or model with BERT layers) with the weights
#' from the given checkpoint.
#'
#' @param mod BERT model (or model with BERT layers) to update with pretrained
#'   weights. Modified in place.
#' @param ckpt_dir Directory of checkpoint with pretrained weights.
#'
#' @return TRUE, invisibly. Function is called for side effects.
#'
#' @keywords internal
.load_checkpoint_weights <- function(mod, ckpt_dir) {
  ckpt_path <- find_ckpt(ckpt_dir)
  ckpt_reader <- tensorflow::tf$train$load_checkpoint(ckpt_path)

  stock_weights_names <- names(ckpt_reader$get_variable_to_dtype_map())
  stock_weights_values <- purrr::map(stock_weights_names,
                                     function(n) {
                                       ckpt_reader$get_tensor(n)
                                     })

  canonical_checkpoint_names <- paste0(stock_weights_names, ":0")

  mwts <- mod$get_weights()

  model_vnames <- purrr::map_chr(mod$variables, "name")

  # Accomodate models with "bert_1", etc. Though I can't get this to happen
  # with TF >= 2.3.
  canonical_model_vnames <- stringr::str_replace_all(model_vnames,
                                                     pattern = "bert(_[\\d]+)?",
                                                     replacement = "bert")
  name_map <- canonical_model_vnames
  names(name_map) <- canonical_model_vnames
  # There are a few exceptions that I can't easily change in RBERT, so just
  # manually adjust them here.
  #TODO: we can save a lot of time here by just saving the name map rather than
  #reconstructing it each time!
  name_map[["bert/embeddings/word_embeddings/embeddings:0"]] <-
    "bert/embeddings/word_embeddings:0"
  name_map[["bert/embeddings/token_type_embeddings/embeddings:0"]] <-
    "bert/embeddings/token_type_embeddings:0"
  name_map[["bert/embeddings/position_embeddings/position_embedding:0"]] <-
    "bert/embeddings/position_embeddings:0"

  # the name_map may have other non-pretrained layers, so keep just the names
  # that overlap:
  bert_layer_name_map <- name_map[name_map %in% canonical_checkpoint_names]

  # Use bert_layer_name_map to copy checkpoint weights by name over into model
  # weights.
  for (vn in names(bert_layer_name_map)) {
    cpn <- bert_layer_name_map[[vn]]

    mwts[names(bert_layer_name_map) == vn][[1]] <-
      stock_weights_values[canonical_checkpoint_names == cpn][[1]]
  }
  # mod is passed by reference, not value, so this sets the weights on the
  # input model.
  mod$set_weights(mwts)
  return(invisible(TRUE))
}


