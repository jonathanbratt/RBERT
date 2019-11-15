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

# These functions are very much in draft form.

# get_params_from_checkpoint ---------------------------------------------

# our version of this function just needs to be able to extract a list
# from a json.
get_params_from_checkpoint <- function(json_file) {
  # /shared/BERT_checkpoints/uncased_L-12_H-768_A-12/bert_config.json
  # bert_ckpt_dir <- "/shared/BERT_checkpoints/uncased_L-12_H-768_A-12/"
  # json_file <- file.path(bert_ckpt_dir, "bert_config.json") #generalize this
  bc <- jsonlite::fromJSON(json_file)

  # some parameters need to be renamed...
  bert_params <- list(
    num_layers = bc$num_hidden_layers,
    num_heads = bc$num_attention_heads,
    hidden_size = bc$hidden_size,
    hidden_dropout = bc$hidden_dropout_prob,
    attention_dropout = bc$attention_probs_dropout_prob,

    intermediate_size = bc$intermediate_size,
    intermediate_activation = bc$hidden_act,

    vocab_size = bc$vocab_size,
    use_token_type = TRUE,
    use_position_embeddings = TRUE,
    token_type_vocab_size = bc$type_vocab_size,
    max_position_embeddings = bc$max_position_embeddings,

    embedding_size = bc$embedding_size,
    shared_layer = !is.null(bc$embedding_size))
  return(bert_params)
}



# get_weights_from_checkpoint --------------------------------------------
#

# load_bert_weights <- function(bert, ckpt_path) {
get_weights_from_checkpoint <- function(ckpt_path) {
  # put checks here to return helpful messages if bad arguments given
  # ckpt_path <- "/shared/BERT_checkpoints/uncased_L-12_H-768_A-12/bert_model.ckpt"

  ckpt_reader <- tensorflow::tf$train$load_checkpoint(ckpt_path)

  stock_weights_names <- names(ckpt_reader$get_variable_to_dtype_map())
  stock_weights_values <- purrr::map(stock_weights_names,
                                     function(n) {
                                       ckpt_reader$get_tensor(n)
                                     })

  names(stock_weights_values) <- stock_weights_names
  return(stock_weights_values)
}

