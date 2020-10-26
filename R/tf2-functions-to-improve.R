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

# functions ----------------------------------------------------------

#' @export
load_bert_model <- function(ckpt_dir, n_token_max = 128L) {
  # keras::k_clear_session() # do this?
  param_list <- get_params_from_checkpoint(ckpt_dir = ckpt_dir)
  input1 <- keras::layer_input(shape = list(n_token_max),
                               dtype = "int32",
                               name = "token_ids")
  # for token type
  input2 <- keras::layer_input(shape = list(n_token_max),
                               dtype = "int32",
                               name = "ttype_ids")

  blayer_out <- custom_layer_BERT(list(input1, input2),
                                  param_list = param_list,
                                  name = "bert",
                                  return_all_layers = TRUE)
  # define the actual model
  model_b <- keras::keras_model(inputs = list(input1, input2),
                                outputs = c(blayer_out$initial_embeddings,
                                            blayer_out$layer_output,
                                            blayer_out$attention_matrix)
  )
  attr(model_b, "ckpt_dir") <- ckpt_dir
  attr(model_b, "n_token_max") <- n_token_max
  attr(model_b, "param_list") <- param_list
  # model_b is basically a pointer to a python object, which is passed by reference.
  .load_checkpoint_weights(model_b, ckpt_dir)

  return(model_b)
}

#' @export
run_bert_model_on_text <- function(bert_model, text) {
  ckpt_dir <- attr(bert_model, "ckpt_dir")
  n_token_max <- attr(bert_model, "n_token_max")
  param_list <- attr(bert_model, "param_list")
  #This doesn't work yet with multi-segment input.
  #TODO: fix that! And make this whole thing better. The tokenization stuff
  # here is just a patch to get this working; needs to be written properly.
  tokenized_text <- tokenize_text(text, ckpt_dir = ckpt_dir)
  # This correctly handles both single and multiple inputs to `text`:
  n_inputs <- length(tokenized_text)
  vocab_file <- find_vocab(ckpt_dir)
  vocab <- load_vocab(vocab_file = vocab_file)
  tids <- purrr::map(tokenized_text, function(tt) {
    tmp <- convert_tokens_to_ids(vocab = vocab, tokens = tt)
    c(tmp, rep(0, n_token_max-length(tmp))) # pad with zeros
  })
  #TODO add check for n_token_max here.
  # Update this to work with multi-segment inputs.
  ttypes <- purrr::map(tokenized_text, function(tt) {
    rep(0, n_token_max)
  })

  bert_input <- list(token_ids = t(array(unlist(tids),
                                         c(n_token_max,n_inputs))),
                     ttype_ids = t(array(unlist(ttypes),
                                         c(n_token_max,n_inputs)))
  )

  bert_out <- bert_model$predict(bert_input)

  emb_out <- .process_embedding_arrays(bert_out, ttypes, tids)
  att_out <- .process_attention_arrays(bert_out, ttypes, tids)

  return(list("output" = emb_out, # maybe we should call this "embeddings"?
              "attention" = att_out))
}

.process_embedding_arrays <- function(bert_out, ttypes, tids) {
  # infer some values from the output...
  d <- dim(bert_out[[1]])
  n_inputs <- d[[1]]
  n_token_max <- d[[2]]
  hidden_size <- d[[3]]
  # assuming we include the zeroth layer embeddings, and attention:
  n_layer <- (length(bert_out) - 1) /2
  segment_index <- unlist(ttypes) + 1 # at this point, make 1-indexed
  token_id_index <- unlist(tids) # maybe should include this in output too?
  token <-  names(token_id_index) # make sure this is robust
  sequence_index <- rep(seq_len(n_inputs), each = n_token_max)
  token_index <- rep(seq_len(n_token_max), times = n_inputs)

  # the first list element in bert_out is the zeroth layer embeddings
  # the next <n_layer> elements are the layer outputs
  # the final <n_layer> elements are the attention matrices for each layer
  big_output <- tibble::tibble(
    sequence_index = integer(),
    segment_index = integer(),
    token_index = integer(),
    token = character(),
    layer_index = integer()
  )

  for (colname in paste0("V", seq_len(hidden_size))) {
    big_output[[colname]] <- integer()
  }

  layer_vec <- 1:(n_layer+1)
  names(layer_vec) <- layer_vec - 1 # purrr::map_df is silly.
  emb_out <- purrr::map_df(layer_vec, function(l) {
    # Reshape array to stack by sequence.
    embmat <- t(array(aperm(bert_out[[l]], c(3,2,1)),
                      c(hidden_size, n_inputs*n_token_max)))
    emb_df <- as.data.frame(embmat)
    emb_df[["segment_index"]] <- segment_index
    emb_df[["token"]] <- token
    emb_df[["layer_index"]] <- rep(l-1, length(token))
    emb_df[["sequence_index"]] <- sequence_index
    emb_df[["token_index"]] <- token_index
    emb_df <- dplyr::filter(emb_df, token != "")
    return(emb_df)
  })
  emb_out <- dplyr::bind_rows(big_output, emb_out)
  return(emb_out)
}

.process_attention_arrays <- function(bert_out, ttypes, tids) {
  # infer some values from the output...
  # assuming we include the zeroth layer embeddings, and attention:
  n_layer <- (length(bert_out) - 1) /2
  d <- dim(bert_out[[n_layer+2]])
  n_inputs <- d[[1]]
  num_heads <- d[[2]]
  n_token_max <- d[[3]]

  segment_index <- unlist(ttypes) + 1 # at this point, make 1-indexed
  token_id_index <- unlist(tids) # maybe should include this in output too?
  token <-  names(token_id_index) # make sure this is robust
  sequence_index <- rep(seq_len(n_inputs), each = n_token_max)
  token_index <- rep(seq_len(n_token_max), times = n_inputs)

  big_attention <- tibble::tibble(
    sequence_index = integer(),
    token_index = integer(),
    segment_index = integer(),
    token = character(),
    layer_index = integer(),
    head_index = integer(),
    attention_token_index = integer(),
    attention_segment_index = integer(),
    attention_token = character(),
    attention_weight = double()
  )

  att_id_cols <- expand.grid(attention_token_index = seq_len(n_token_max),
                             token_index = seq_len(n_token_max),
                             head_index = seq_len(num_heads),
                             sequence_index = seq_len(n_inputs))
  # joining on (twice) is the best way to add tokens, segments? Make dataframe
  # with column for sequence_index, token_index, token and segment_index.

  att_labels <- tibble::tibble(
    token_index = token_index,
    sequence_index = sequence_index,
    token = token,
    segment_index = segment_index
  )
  att_id_cols <- dplyr::left_join(att_id_cols, att_labels,
                                  by = c("attention_token_index" = "token_index",
                                         "sequence_index" = "sequence_index"))
  att_id_cols <- dplyr::rename(att_id_cols,
                               "attention_token" = token,
                               "attention_segment_index" = segment_index)
  att_id_cols <- dplyr::left_join(att_id_cols, att_labels,
                                  by = c("token_index" = "token_index",
                                         "sequence_index" = "sequence_index"))

  layer_vec <- 1:n_layer # no zeroth layer for attention
  names(layer_vec) <- layer_vec # purrr::map_df is silly.
  att_out <- purrr::map_df(layer_vec, function(l) {
    # Reshape array to single column, indexed by (fastest to slowest):
    # attention_to_token, attention_from_token, head_index, sequence_index.
    # Attention outputs start after n_layer + 1 elements of embeddings output.
    attmat <- t(array(aperm(bert_out[[l + n_layer + 1]],
                            c(4,3,2,1)),
                      c(1,d[[4]]*d[[3]]*d[[2]]*d[[1]])))
    att_df <- as.data.frame(attmat)
    att_df <- dplyr::rename(att_df, "attention_weight" = V1)
    att_df <- dplyr::bind_cols(att_id_cols, att_df)
    att_df <- dplyr::filter(att_df, token != "", attention_token != "")
    att_df <- dplyr::mutate(att_df, layer_index = l)
  })
  att_out <- dplyr::bind_rows(big_attention, att_out)

}



