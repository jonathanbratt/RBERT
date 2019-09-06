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


# BertConfig --------------------------------------------------------------------

#' Construct objects of BertConfig class
#'
#' Given a set of values as parameter inputs, construct a BertConfig object with
#' those values.
#'
#' @param vocab_size Integer; vocabulary size of \code{inputs_ids} in
#'   \code{BertModel}.
#' @param hidden_size Integer; size of the encoder layers and the pooler layer.
#' @param num_hidden_layers Integer; number of hidden layers in the Transformer
#'   encoder.
#' @param num_attention_heads Integer; number of attention heads for each
#'   attention layer in the Transformer encoder.
#' @param intermediate_size Integer; the size of the "intermediate" (i.e.,
#'   feed-forward) layer in the Transformer encoder.
#' @param hidden_act The non-linear activation function (function or string) in
#'   the encoder and pooler.
#' @param hidden_dropout_prob Numeric; the dropout probability for all fully
#'   connected layers in the embeddings, encoder, and pooler.
#' @param attention_probs_dropout_prob Numeric; the dropout ratio for the
#'   attention probabilities.
#' @param max_position_embeddings Integer; the maximum sequence length that this
#'   model might ever be used with. Typically set this to something large just
#'   in case (e.g., 512 or 1024 or 2048).
#' @param type_vocab_size Integer; the vocabulary size of the
#'   \code{token_type_ids} passed into \code{BertModel}.
#' @param initializer_range Numeric; the stdev of the
#'   truncated_normal_initializer for initializing all weight matrices.
#'
#' @return An object of class BertConfig
#' @export
#'
#' @examples
#' \dontrun{
#' BertConfig(vocab_size = 30522L)
#' }
BertConfig <- function(vocab_size,
                       hidden_size = 768L,
                       num_hidden_layers = 12L,
                       num_attention_heads = 12L,
                       intermediate_size = 3072L,
                       hidden_act = "gelu",
                       hidden_dropout_prob = 0.1,
                       attention_probs_dropout_prob = 0.1,
                       max_position_embeddings = 512L,
                       type_vocab_size = 16L,
                       initializer_range = 0.02) {
  obj <- list("vocab_size" = vocab_size,
              "hidden_size" = hidden_size,
              "num_hidden_layers" = num_hidden_layers,
              "num_attention_heads" = num_attention_heads,
              "hidden_act" = hidden_act,
              "intermediate_size" = intermediate_size,
              "hidden_dropout_prob" = hidden_dropout_prob,
              "attention_probs_dropout_prob" = attention_probs_dropout_prob,
              "max_position_embeddings" = max_position_embeddings,
              "type_vocab_size" = type_vocab_size,
              "initializer_range" = initializer_range)
  class(obj) <- "BertConfig"
  return(obj)
}

# For RBERT, only implement the json config reader. And don't bother with
# the class stuff... but then this all looks pretty ridiculous. Keep
# for now; refactor later.

#' Load BERT config object from json file
#'
#' Given a path to a json config file, construct a BertConfig object with
#' appropriate values.
#'
#' @param json_file Character; the path to a json config file.
#'
#' @return An object of class BertConfig
#' @export
#'
#' @examples
#' \dontrun{
#' json_file <- file.path("/shared",
#'                        "BERT_checkpoints",
#'                        "uncased_L-12_H-768_A-12",
#'                        "bert_config.json")
#' bert_config_from_json_file(json_file)
#' }
bert_config_from_json_file <- function(json_file) {
  args <- jsonlite::fromJSON(json_file)
  return(do.call(BertConfig, args = args))
}


# BertModel ---------------------------------------------------------------

#' Construct object of class BertModel
#'
#' An object of class BertModel has several elements:
#' \describe{
#' \item{embedding_output}{float Tensor of shape \code{[batch_size, seq_length,
#' hidden_size]} corresponding to the output of the embedding layer, after
#' summing the word embeddings with the positional embeddings and the token type
#' embeddings, then performing layer normalization. This is the input to the
#' transformer.}
#' \item{embedding_table}{The table for the token embeddings.}
#' \item{all_encoder_layers}{A list of float Tensors of shape \code{[batch_size,
#' seq_length, hidden_size]}, corresponding to all the hidden transformer
#' layers.}
#' \item{sequence_output}{float Tensor of shape \code{[batch_size, seq_length,
#' hidden_size]} corresponding to the final hidden layer of the transformer
#' encoder.}
#' \item{pooled_output}{The dense layer on top of the hidden layer for the first
#' token.}
#' }
#'
#'
#' @param config \code{BertConfig} instance.
#' @param is_training Logical; TRUE for training model, FALSE for eval model.
#'   Controls whether dropout will be applied.
#' @param input_ids Int32 Tensor of shape \code{[batch_size, seq_length]}.
#' @param input_mask (optional) Int32 Tensor of shape \code{[batch_size,
#'   seq_length]}.
#' @param token_type_ids (optional) Int32 Tensor of shape \code{[batch_size,
#'   seq_length]}.
#' @param use_one_hot_embeddings (optional) Logical; whether to use one-hot word
#'   embeddings or tf.embedding_lookup() for the word embeddings.
#' @param scope (optional) Character; name for variable scope. Defaults to
#'   "bert".
#'
#' @return An object of class BertModel.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      {
#'        input_ids <- tensorflow::tf$constant(list(list(31L, 51L, 99L),
#'                                                  list(15L, 5L, 0L)))
#'
#'        input_mask <- tensorflow::tf$constant(list(list(1L, 1L, 1L),
#'                                                   list(1L, 1L, 0L)))
#'        token_type_ids <- tensorflow::tf$constant(list(list(0L, 0L, 1L),
#'                                                       list(0L, 2L, 0L)))
#'        config <- BertConfig(vocab_size = 32000L,
#'                             hidden_size = 768L,
#'                             num_hidden_layers = 8L,
#'                             num_attention_heads = 12L,
#'                             intermediate_size = 1024L)
#'        model <- BertModel(config = config,
#'                          is_training = TRUE,
#'                           input_ids = input_ids,
#'                           input_mask = input_mask,
#'                           token_type_ids = token_type_ids)
#'      }
#' )
#' }
BertModel <- function(config,
                      is_training,
                      input_ids,
                      input_mask = NULL,
                      token_type_ids = NULL,
                      use_one_hot_embeddings = FALSE,
                      scope = NULL) {
  if (!is_training) {
    config$hidden_dropout_prob <- 0.0
    config$attention_probs_dropout_prob <- 0.0
  }

  input_shape <- get_shape_list(input_ids, expected_rank = 2L)
  batch_size <- input_shape[[1]]
  seq_length <- input_shape[[2]]

  if (is.null(input_mask)) {
    input_mask  <- tensorflow::tf$ones(
      shape = tensorflow::shape(batch_size, seq_length),
      dtype = tensorflow::tf$int32)
  }

  if (is.null(token_type_ids)) {
    token_type_ids <- tensorflow::tf$zeros(
      shape = tensorflow::shape(batch_size, seq_length),
      dtype = tensorflow::tf$int32)
  }

  with(tensorflow::tf$variable_scope(scope, default_name = "bert"), {
    with(tensorflow::tf$variable_scope("embeddings"), {
      # Perform embedding lookup on the word ids.
      elup <- embedding_lookup(
        input_ids = input_ids,
        vocab_size = config$vocab_size,
        embedding_size = config$hidden_size,
        initializer_range = config$initializer_range,
        word_embedding_name = "word_embeddings",
        use_one_hot_embeddings = use_one_hot_embeddings)
      embedding_output <- elup[[1]]
      embedding_table <- elup[[2]]

      # Add positional embeddings and token type embeddings, then layer
      # normalize and perform dropout.
      embedding_output <-  embedding_postprocessor(
        input_tensor = embedding_output,
        use_token_type = TRUE,
        token_type_ids = token_type_ids,
        token_type_vocab_size = config$type_vocab_size,
        token_type_embedding_name = "token_type_embeddings",
        use_position_embeddings = TRUE,
        position_embedding_name = "position_embeddings",
        initializer_range = config$initializer_range,
        max_position_embeddings = config$max_position_embeddings,
        dropout_prob = config$hidden_dropout_prob)
    })

    with(tensorflow::tf$variable_scope("encoder"), {
      # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
      # mask of shape [batch_size, seq_length, seq_length] which is used
      # for the attention scores.
      attention_mask <- create_attention_mask_from_input_mask(input_ids,
                                                              input_mask)

      # Run the stacked transformer.
      # `sequence_output` shape = [batch_size, seq_length, hidden_size].
      all_encoder_layers <- transformer_model(
        input_tensor = embedding_output,
        attention_mask = attention_mask,
        hidden_size = config$hidden_size,
        num_hidden_layers = config$num_hidden_layers,
        num_attention_heads = config$num_attention_heads,
        intermediate_size = config$intermediate_size,
        intermediate_act_fn = get_activation(config$hidden_act),
        hidden_dropout_prob = config$hidden_dropout_prob,
        attention_probs_dropout_prob = config$attention_probs_dropout_prob,
        initializer_range = config$initializer_range,
        do_return_all_layers = TRUE)
    })

    # ATTN: modified below to separate out attention_data
    attention_data <- all_encoder_layers$attention_data
    all_encoder_layers <- all_encoder_layers$final_outputs
    # ATTN: modified above to separate out attention_data

    sequence_output <- all_encoder_layers[[length(all_encoder_layers)]]

    # The "pooler" converts the encoded sequence tensor of shape
    # [batch_size, seq_length, hidden_size] to a tensor of shape
    # [batch_size, hidden_size]. This is necessary for segment-level
    # (or segment-pair-level) classification tasks where we need a fixed
    # dimensional representation of the segment.
    with(tensorflow::tf$variable_scope("pooler"), {
      # We "pool" the model by simply taking the hidden state corresponding
      # to the first token. We assume that this has been pre-trained.

      # Why the first token? I wondered, too. There's a nice explanation here:
      # https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/#research-finding-an-effective-embeddingencoding
      # tl;dr: The first token is the '[CLS]' token. All the layers
      # corresponding to this token can (plausibly) contain enough of the
      # context from the entire input to be used for classification. It is far
      # from obvious that this is the optimal approach, but at least it's
      # reasonable. -JDB

      # This is one of those places where we use 1-based indexing. -JDB
      # "The (R) tensorflow package now uses 1-based extraction by default."
      first_token_tensor <- sequence_output[, 1, ]
      pooled_output <- tensorflow::tf$layers$dense(
        first_token_tensor,
        config$hidden_size,
        activation = tensorflow::tf$tanh,
        kernel_initializer = create_initializer(config$initializer_range))
    })
  })
  obj <- list("embedding_output" = embedding_output,
              "embedding_table" = embedding_table,
              "all_encoder_layers" = all_encoder_layers,
              # ATTN: modified below to include attention_data in output
              "attention_data" = attention_data,
              # ATTN: modified above to include attention_data in output
              "sequence_output" = sequence_output,
              "pooled_output" = pooled_output)
  class(obj) <- "BertModel"
  return(obj)
}

# The python "BertModel" class had methods for extracting the object elements
# (e.g. get_pooled_output, get_sequence_output). These methods seem to be
# unneeded in this implementation. It's just as easy to do:
# `model$pooled_output`, etc.

# gelu --------------------------------------------------------------------

#' Gaussian Error Linear Unit
#'
#' This is a smoother version of the RELU. Original paper:
#' https://arxiv.org/abs/1606.08415
#'
#' @param x Float Tensor to perform activation on.
#'
#' @return `x` with the GELU activation applied.
#' @export
#'
#' @examples
#' \dontrun{
#' tfx <- tensorflow::tf$get_variable("none", tensorflow::shape(10L))
#' gelu(tfx)
#' }
gelu <- function(x) {
  cdf <- 0.5*(1.0 + tensorflow::tf$tanh(
    (sqrt(2/pi)*(x + 0.044715 * tensorflow::tf$pow(x, 3))))
  )
  return(x*cdf)
}


# get_activation ----------------------------------------------------------

#' Map a string to a Python function
#'
#' Example: "relu" => `tensorflow::tf$nn$relu`.
#'
#' @param activation_string String name of the activation function.
#'
#' @return A function corresponding to the activation function. If
#'   \code{activation_string} is NA, empty, or "linear", this will return NA. If
#'   \code{activation_string} is not a string, it will return
#'   \code{activation_string}.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' get_activation("gelu")
#' }
get_activation <- function(activation_string) {
  if (!is.character(activation_string)) {
    # this is the python behavior, but I think I should return this only if
    # activation_string has class "function" or "python.builtin.function",
    # and return NA otherwise.
    return(activation_string)
  }
  activation_string <- tolower(trimws(activation_string))
  if (is.na(activation_string) | activation_string=="") {
    return(NA)
  }
  # if we add dplyr to imports, change this to a case_when?
  if (activation_string == "linear") {
    return(NA)
  } else if (activation_string == "relu") {
    return(tensorflow::tf$nn$relu)
  } else if (activation_string == "gelu") {
    return(gelu)
  } else if (activation_string == "tanh") {
    return(tensorflow::tf$tanh)
  } else {
    stop(paste("Unsupported activation", activation_string))
  }
}



# get_assignment_map_from_checkpoint ------------------------------------------

#' Compute the intersection of the current variables and checkpoint variables
#'
#' Returns the intersection (not the union, as python docs say -JDB) of the sets
#' of variable names from the current graph and the checkpoint.
#'
#' Note that a Tensorflow checkpoint is not the same as a saved model. A saved
#' model contains a complete description of the computational graph and is
#' sufficient to reconstruct the entire model, while a checkpoint contains just
#' the parameter values (and variable names), and so requires a specification of
#' the original model structure to reconstruct the computational graph. -JDB
#'
#' @param tvars List of training variables in the current model.
#' @param init_checkpoint Character; path to the checkpoint directory, plus
#'   checkpoint name stub (e.g. "bert_model.ckpt"). Path must be absolute and
#'   explicit, starting with "/".
#'
#' @return List with two elements: the assignment map and the initialized
#'   variable names. The assignment map is a list of the "base" variable names
#'   that are in both the current computational graph and the checkpoint. The
#'   initialized variable names list contains both the base names and the base
#'   names + ":0". (This seems redundant to me. I assume it will make sense
#'   later. -JDB)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Just for illustration: create a "model" with a couple variables
#' # that overlap some variable names in the BERT checkpoint.
#' with(tensorflow::tf$variable_scope("bert",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      {
#'        test_ten1 <- tensorflow::tf$get_variable(
#'          "encoder/layer_9/output/dense/bias",
#'          shape = c(1L, 2L, 3L)
#'        )
#'        test_ten2 <- tensorflow::tf$get_variable(
#'          "encoder/layer_9/output/dense/kernel",
#'          shape = c(1L, 2L, 3L)
#'        )
#'      }
#' )
#' tvars <- tensorflow::tf$get_collection(
#'   tensorflow::tf$GraphKeys$GLOBAL_VARIABLES
#' )
#'
#' init_checkpoint <- file.path("/shared",
#'                              "BERT_checkpoints",
#'                              "uncased_L-12_H-768_A-12",
#'                              "bert_model.ckpt")
#'
#' amap <- get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#' }
get_assignment_map_from_checkpoint <- function(tvars, init_checkpoint) {
  # I think this function could be streamlined a lot, but I'll save
  # that for the eventual refactor. (Like, why create name_to_variable?
  # It seems to be used just as a list of names; nothing is done with
  # the variables.) -JDB
  assignment_map <- list()
  initialized_variable_names <- list()

  name_to_variable <- list()
  for (var in tvars) {
    name <- var$name
    # extract the base name: the part up to the last :<number>
    m <- stringr::str_match(string = name, pattern = "^(.*):\\d+$")[[2]]
    if (!is.na(m)) {
      name <- m
    }
    name_to_variable[[name]] <- var
  }
  just_names <- names(name_to_variable)

  init_vars <- tensorflow::tf$train$list_variables(init_checkpoint)

  assignment_map <- list()

  for (x in init_vars) {
    name <- x[[1]]
    if (name %in% just_names) {
      assignment_map[[name]] <- name #why?
      initialized_variable_names[[name]] <- 1
      initialized_variable_names[[paste0(name, ":0")]] <- 1
    }
  }
  return(list("assignment_map" = assignment_map,
              "initialized_variable_names" = initialized_variable_names))
}

# dropout --------------------------------------------------------------------

#' Perform Dropout
#'
#' @param input_tensor Float Tensor to perform dropout on.
#' @param dropout_prob A double giving the probability of dropping out a value
#'   (NOT of KEEPING a dimension as in `tf.nn.dropout`).
#'
#' @return A version of `input_tensor` with dropout applied.
#' @export
#'
#' @examples
#' \dontrun{
#' tfx <- tensorflow::tf$get_variable("none", tensorflow::shape(10L))
#' dropout(tfx, 0.5)
#' }
dropout <- function(input_tensor, dropout_prob = NULL) {
  if (is.null(dropout_prob) | dropout_prob == 0.0) {
    return(input_tensor)
  }
  output <- tensorflow::tf$nn$dropout(input_tensor, 1.0 - dropout_prob)
  return(output)
}


# layer_norm --------------------------------------------------------------------

#' Run layer normalization
#'
#' Run layer normalization on the last dimension of the tensor.
#'
#' Wrapper around tensorflow layer_norm function. From tensorflow documentation:
#' Adds a Layer Normalization layer. Based on the paper:
#' \url{https://arxiv.org/abs/1607.06450}.
#'
#' Note: \code{begin_norm_axis}: The first normalization dimension:
#' normalization will be performed along dimensions (begin_norm_axis :
#' rank(inputs) )
#'
#' \code{begin_params_axis}: The first parameter (beta, gamma) dimension: scale
#' and centering parameters will have dimensions (begin_params_axis :
#' rank(inputs) ) and will be broadcast with the normalized inputs accordingly.
#'
#' @param input_tensor  Tensor to perform layor normalization on.
#' @param name Optional variable_scope for layer_norm.
#'
#' @return A Tensor of the same shape and type as `input_tensor`, with
#'   normalization applied.
#' @export
#'
#' @examples
#' \dontrun{
#' tfx <- tensorflow::tf$get_variable("example", tensorflow::shape(10L))
#' layer_norm(tfx)
#' }
layer_norm <- function(input_tensor, name = NULL) {
  return(tensorflow::tf$contrib$layers$layer_norm(
    inputs = input_tensor,
    begin_norm_axis = -1L,
    begin_params_axis = -1L,
    scope = name))
}


# layer_norm_and_dropout --------------------------------------------------------------------

#' Run layer normalization followed by dropout
#'
#' @param input_tensor Float Tensor to perform layer_norm and dropout on.
#' @param dropout_prob A double describing the probability of dropping out a
#'   value (NOT of KEEPING a dimension as in `tf.nn.dropout`).
#' @param name Optional variable_scope for layer_norm.
#'
#' @return Tensor resulting from applying layer_norm and dropout to
#'   \code{input_tensor}.
#' @export
#'
#' @examples
#' \dontrun{
#' tfx <- tensorflow::tf$get_variable("example2", tensorflow::shape(10L))
#' layer_norm_and_dropout(tfx, dropout_prob = 0.5)
#' }
layer_norm_and_dropout <- function(input_tensor,
                                   dropout_prob = NULL,
                                   name = NULL) {
  output_tensor <- layer_norm(input_tensor, name)
  output_tensor <- dropout(output_tensor, dropout_prob)
  return(output_tensor)
}

# create_initializer ----------------------------------------------------------

#' Create truncated normal initializer
#'
#' This is a wrapper around the tensorflow truncated_normal_initializer
#' function.
#'
#' @param initializer_range A double describing the range for the initializer
#'   (passed to the stddev parameter).
#'
#' @return A tensorflow initializer.
#' @export
#'
#' @examples
#' \dontrun{
#' create_initializer(0.02)
#' }
create_initializer <- function(initializer_range = 0.02) {
  return(
    tensorflow::tf$truncated_normal_initializer(stddev = initializer_range)
  )
}

# embedding_lookup  ----------------------------------------------------------

#' Look up words embeddings for id tensor
#'
#' @param input_ids Integer Tensor of shape [batch_size, seq_length] containing
#'   word ids.
#' @param vocab_size Size of the embedding vocabulary (integer).
#' @param embedding_size Width of the word embeddings (integer).
#' @param initializer_range Embedding initialization range (float).
#' @param word_embedding_name Name of the embedding table (character).
#' @param use_one_hot_embeddings If TRUE, use one-hot method for word
#'   embeddings. If FALSE, use \code{tf$gather()}.
#'
#' @return Float Tensor of shape [batch_size, seq_length, embedding_size], along
#'   with the embedding table in a list.
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      ids <- tensorflow::tf$get_variable("x", dtype = "int32",
#'                                         shape = tensorflow::shape(10, 20))
#' )
#' embedding_lookup(ids, vocab_size = 100, word_embedding_name = "some_name")
#'
#' }
embedding_lookup <- function(input_ids,
                             vocab_size,
                             embedding_size = 128L,
                             initializer_range = 0.02,
                             word_embedding_name = "word_embeddings",
                             use_one_hot_embeddings = FALSE) {
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if (input_ids$shape$ndims == 2L) {
    input_ids <- tensorflow::tf$expand_dims(input_ids, axis = c(-1L))
  } else if (input_ids$shape$ndims < 2 | input_ids$shape$ndims > 3) {
    stop("input_id tensor has incorrect shape.")
  }

  embedding_table <-  tensorflow::tf$get_variable(
    name = word_embedding_name,
    shape = tensorflow::shape(vocab_size, embedding_size),
    initializer = create_initializer(initializer_range))

  # The shape argument can usually be a vector, but that fails if it's
  # a single-element vector. So use list in this case.
  flat_input_ids <- tensorflow::tf$reshape(input_ids, shape = list(-1L))

  if (use_one_hot_embeddings) {
    one_hot_input_ids <- tensorflow::tf$one_hot(flat_input_ids,
                                                depth = as.integer(vocab_size))
    output <- tensorflow::tf$matmul(one_hot_input_ids, embedding_table)
  } else {
    output <- tensorflow::tf$gather(embedding_table, flat_input_ids)
  }

  input_shape <- unlist(get_shape_list(input_ids))
  num_dims <- length(input_shape)
  last_dim <- input_shape[num_dims][[1]]
  first_dims <- input_shape[-num_dims]
  target_shape <- unlist(list(first_dims, last_dim*embedding_size),
                         recursive = FALSE)
  output <-  tensorflow::tf$reshape(output,
                                    target_shape)

  return(list(output, embedding_table))
}


# embedding_postprocessor  ----------------------------------------------------

#' Perform various post-processing on a word embedding tensor
#'
#' This function (optionally) adds to the word embeddings additional embeddings
#' for token type and position.
#'
#' See figure 2 in the BERT paper:
#'
#' \url{https://arxiv.org/pdf/1810.04805.pdf}
#'
#' Both type and position embeddings are learned model variables. Note that
#' token "type" is essentially a sentence identifier, indicating which sentence
#' (or, more generally, piece of text) the token belongs to.
#'
#' @param input_tensor Float Tensor of shape \code{[batch_size, seq_length,
#'   embedding_size]}.
#' @param use_token_type Logical; whether to add embeddings for
#'   \code{token_type_ids}.
#' @param token_type_ids (optional) Integer Tensor of shape \code{[batch_size,
#'   seq_length]}. Must be specified if \code{use_token_type} is TRUE
#' @param token_type_vocab_size Integer; the vocabulary size of
#'   \code{token_type_ids}. This defaults to 16 (here and in BERT code), but
#'   must be set to 2 for compatibility with saved checkpoints.
#' @param token_type_embedding_name Character; the name of the embedding table
#'   variable for token type ids.
#' @param use_position_embeddings Logical; whether to add position embeddings
#'   for the position of each token in the sequence.
#' @param position_embedding_name Character; the name of the embedding table
#'   variable for positional embeddings.
#' @param initializer_range Numeric; range of the weight initialization.
#' @param max_position_embeddings Integer; maximum sequence length that might
#'   ever be used with this model. This can be longer than the sequence length
#'   of input_tensor, but cannot be shorter.
#' @param dropout_prob Numeric; dropout probability applied to the final output
#'   tensor.
#'
#' @return Float Tensor with same shape as \code{input_tensor}.
#' @export
#'
#' @examples
#' \dontrun{
#' batch_size <- 10
#' seq_length <- 512
#' embedding_size <- 200
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      {
#'        input_tensor <- tensorflow::tf$get_variable(
#'          "input", dtype = "float",
#'          shape = tensorflow::shape(batch_size, seq_length, embedding_size))
#'        token_type_ids <- tensorflow::tf$get_variable(
#'          "ids", dtype = "int32",
#'          shape = tensorflow::shape(batch_size, seq_length))
#'      }
#' )
#' embedding_postprocessor(input_tensor,
#'                         use_token_type = TRUE,
#'                         token_type_ids = token_type_ids)
#' }
embedding_postprocessor <- function(
  input_tensor,
  use_token_type = FALSE,
  token_type_ids = NULL,
  token_type_vocab_size = 16L,
  token_type_embedding_name = "token_type_embeddings",
  use_position_embeddings = TRUE,
  position_embedding_name = "position_embeddings",
  initializer_range = 0.02,
  max_position_embeddings = 512L,
  dropout_prob = 0.1
) {
  input_shape <-  get_shape_list(input_tensor, expected_rank = 3L)
  batch_size <- input_shape[[1]]
  seq_length <- input_shape[[2]]
  width <- input_shape[[3]] # a.k.a. embedding_size? -JDB

  output <- input_tensor

  if (use_token_type) {
    if (is.null(token_type_ids)) {
      stop("`token_type_ids` must be specified if `use_token_type` is TRUE.")
    }
    token_type_table <- tensorflow::tf$get_variable(
      name = token_type_embedding_name,
      shape = tensorflow::shape(token_type_vocab_size, width),
      initializer = create_initializer(initializer_range)
    )
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
  }
  flat_token_type_ids <- tensorflow::tf$reshape(token_type_ids,
                                                shape = list(-1L))
  one_hot_ids <- tensorflow::tf$one_hot(flat_token_type_ids,
                                        depth = token_type_vocab_size)
  token_type_embeddings <-  tensorflow::tf$matmul(one_hot_ids,
                                                  token_type_table)
  token_type_embeddings <- tensorflow::tf$reshape(
    token_type_embeddings,
    shape = list(batch_size, seq_length, width)
  )
  output <- output + token_type_embeddings

  if (use_position_embeddings) {
    assert_op <- tensorflow::tf$assert_less_equal(seq_length,
                                                  max_position_embeddings)
    # Double check that list is necessary here.
    with(tensorflow::tf$control_dependencies(list(assert_op)) , {
      # what follows runs only after `assert_op`; see:
      # https://devdocs.io/tensorflow~python/tf/graph#control_dependencies

      full_position_embeddings <- tensorflow::tf$get_variable(
        name = position_embedding_name,
        shape = tensorflow::shape(max_position_embeddings, width),
        initializer = create_initializer(initializer_range)
      )

      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.

      # This is effectively using `slice` to truncate the 0th dimension
      # of the full tensor to have a size of seq_length, rather than
      # max_position_embeddings. -JDB

      position_embeddings <-  tensorflow::tf$slice(
        full_position_embeddings,
        begin = tensorflow::shape(0, 0),
        size = tensorflow::shape(seq_length, -1)
      )

      # This will typically be just 3. -JDB
      num_dims <-  length(output$shape$as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size. (Is it ever anything else? -JDB)

      # This gives `position_embeddings` a shape compatible with `output`.
      # The "extra" dimensions required (which is typically just the
      # first: the batch size) have length one, so the embeddings tensor
      # is simply broadcast (recycled). -JDB
      position_broadcast_shape <- as.integer(c(
        rep.int(1, num_dims - 2),
        seq_length,
        width
      ))
      position_embeddings <- tensorflow::tf$reshape(position_embeddings,
                                                    position_broadcast_shape)
      output <- output + position_embeddings
    })
  }
  output <- layer_norm_and_dropout(output, dropout_prob)
  return(output)
}

# create_attention_mask_from_input_mask  --------------------------------------

#' Create 3D attention mask from a 2D tensor mask
#'
#' An attention mask is used to zero out specific elements of an attention
#' matrix. (For example, to prevent the model from "paying attention to the
#' answer" in certain training tasks.)
#'
#' @param from_tensor 2D or 3D Tensor of shape [batch_size, from_seq_length,
#'   ...].
#' @param to_mask int32 Tensor of shape [batch_size, to_seq_length].
#'
#' @return float Tensor of shape [batch_size, from_seq_length, to_seq_length].
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      {
#'        from_tensor <- ids <- tensorflow::tf$get_variable("ften",
#'                                          dtype = "float", shape = c(10, 20))
#'        to_mask <- ids <- tensorflow::tf$get_variable("mask",
#'                                          dtype = "int32", shape = c(10, 30))
#'      }
#' )
#' create_attention_mask_from_input_mask(from_tensor, to_mask)
#' }
create_attention_mask_from_input_mask <- function(from_tensor, to_mask) {
  from_shape <- get_shape_list(from_tensor, expected_rank = list(2,3))
  batch_size <- from_shape[[1]]
  from_seq_length <- from_shape[[2]]

  to_shape <- get_shape_list(to_mask, expected_rank = 2)
  to_batch_size <- to_shape[[1]]
  to_seq_length <- to_shape[[2]]

  to_mask <- tensorflow::tf$cast(
    tensorflow::tf$reshape(to_mask,
                           list(batch_size, 1L, to_seq_length)),
    tensorflow::tf$float32
  )

  # "We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to*
  # padding tokens) so we create a tensor of all ones."

  broadcast_ones <- tensorflow::tf$ones(
    shape = list(batch_size, from_seq_length, 1L),
    dtype = tensorflow::tf$float32
  )

  # I checked, and this appears to handle the broadcast over tensorflow
  # tensors correctly in R. -JDB
  mask <- broadcast_ones*to_mask
  return(mask)
}

# transpose_for_scores ----------------------------------------------------

#' Reshape and transpose tensor
#'
#' In Python code, this is internal to attention_layer. Pulling it out into
#' separate function here.
#'
#' @param input_tensor Tensor to reshape and transpose.
#' @param batch_size Size of the first dimension of input_tensor.
#' @param num_attention_heads Size of the third dimension of input_tensor. (Will
#'   be transposed to second dimension.)
#' @param seq_length Size of the second dimension of input_tensor. (Will be
#'   transposed to third dimension.)
#' @param width Size of fourth dimension of input_tensor.
#'
#' @return  Tensor of shape: batch_size, num_attention_heads, seq_length, width.
#'
#' @keywords internal
transpose_for_scores <- function(input_tensor,
                                 batch_size,
                                 num_attention_heads,
                                 seq_length,
                                 width) {
  output_tensor <-  tensorflow::tf$reshape(
    input_tensor,
    # We can't use shape() here, because batch_size is still undetermined
    # at this point. -JDB
    list(batch_size,
         as.integer(seq_length),
         as.integer(num_attention_heads),
         as.integer(width))
  )
  # The R tensorflow package indexes from 1 in some places,
  # but the perm parameter labels the dimensions using zero-indexing. *shrug*
  output_tensor <- tensorflow::tf$transpose(output_tensor,
                                            perm = list(0L, 2L, 1L, 3L))
  return(output_tensor)
}

# attention_layer  ----------------------------------------------------------

#' Build multi-headed attention layer
#'
#' Performs multi-headed attention from \code{from_tensor} to \code{to_tensor}.
#' This is an implementation of multi-headed attention based on "Attention is
#' all you Need". If \code{from_tensor} and \code{to_tensor} are the same, then
#' this is self-attention. Each timestep in \code{from_tensor} attends to the
#' corresponding sequence in \code{to_tensor}, and returns a fixed-with vector.
#' This function first projects \code{from_tensor} into a "query" tensor and
#' \code{to_tensor} into "key" and "value" tensors. These are (effectively) a
#' list of tensors of length \code{num_attention_heads}, where each tensor is of
#' shape \code{[batch_size, seq_length, size_per_head]}. Then, the query and key
#' tensors are dot-producted and scaled. These are softmaxed to obtain attention
#' probabilities. The value tensors are then interpolated by these
#' probabilities, then concatenated back to a single tensor and returned.
#'
#' In practice, the multi-headed attention are done with transposes and reshapes
#' rather than actual separate tensors.
#'
#' @param from_tensor Float Tensor of shape \code{[batch_size, from_seq_length,
#'   from_width]}.
#' @param to_tensor Float Tensor of shape \code{[batch_size, to_seq_length,
#'   to_width]}.
#' @param attention_mask (optional) Integer Tensor of shape \code{[batch_size,
#'   from_seq_length, to_seq_length]}. The values should be 1 or 0. The
#'   attention scores will effectively be set to -infinity for any positions in
#'   the mask that are 0, and will be unchanged for positions that are 1.
#' @param num_attention_heads Integer; number of attention heads.
#' @param size_per_head Integer; size of each attention head.
#' @param query_act (Optional) Activation function for the query transform.
#' @param key_act (Optional) Activation function for the key transform.
#' @param value_act (Optional) Activation function for the value transform.
#' @param attention_probs_dropout_prob (Optional) Numeric; dropout probability
#'   of the attention probabilities.
#' @param initializer_range Numeric; range of the weight initializer.
#' @param do_return_2d_tensor Logical. If TRUE, the output will be of shape
#'   \code{[batch_size * from_seq_length, num_attention_heads * size_per_head]}.
#'   If false, the output will be of shape \code{[batch_size, from_seq_length,
#'   num_attention_heads * size_per_head]}.
#' @param batch_size (Optional) Integer; if the input is 2D, this might (sic) be
#'   the batch size of the 3D version of the \code{from_tensor} and
#'   \code{to_tensor}.
#' @param from_seq_length (Optional) Integer; if the input is 2D, this might be
#'   the seq length of the 3D version of the \code{from_tensor}.
#' @param to_seq_length (Optional) Integer; if the input is 2D, this might be
#'   the seq length of the 3D version of the \code{to_tensor}.
#'
#' @return float Tensor of shape \code{[batch_size, from_seq_length,
#'   num_attention_heads * size_per_head]}. If \code{do_return_2d_tensor} is
#'   TRUE, it will be flattened to shape \code{[batch_size * from_seq_length,
#'   num_attention_heads * size_per_head]}.
#' @export
#'
#' @examples
#' \dontrun{
#' # Maybe add examples later. For now, this is only called from
#' # within transformer_model(), so refer to that function.
#' }
attention_layer <- function(from_tensor,
                            to_tensor,
                            attention_mask = NULL,
                            num_attention_heads = 1L,
                            size_per_head = 512L,
                            query_act = NULL,
                            key_act = NULL,
                            value_act = NULL,
                            attention_probs_dropout_prob = 0.0,
                            initializer_range = 0.02,
                            do_return_2d_tensor = FALSE,
                            batch_size = NULL,
                            from_seq_length = NULL,
                            to_seq_length = NULL) {
  from_shape <- get_shape_list(from_tensor, expected_rank = c(2L, 3L))
  to_shape <- get_shape_list(to_tensor, expected_rank = c(2L, 3L))

  if (length(from_shape) != length(to_shape)) {
    stop("The rank of from_tensor must match the rank of to_tensor.")
  }

  # from_shape and to_shape will both have the same length == 2
  # or length == 3 at this point.
  if (length(from_shape) == 3) {
    batch_size <- from_shape[[1]]
    from_seq_length <- from_shape[[2]]
    to_seq_length <- to_shape[[2]]
  } else if (
    is.null(batch_size) | is.null(from_seq_length) | is.null(to_seq_length)
  ) {
    stop(paste("When passing in rank 2 tensors to attention_layer, the values",
               "for batch_size, from_seq_length, and to_seq_length",
               "must all be specified."
    ))
  }

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`
  from_tensor_2d <- reshape_to_matrix(from_tensor)
  to_tensor_2d <- reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer <- tensorflow::tf$layers$dense(
    from_tensor_2d,
    num_attention_heads * size_per_head,
    activation = query_act,
    name = "query",
    kernel_initializer = create_initializer(initializer_range)
  )

  # `key_layer` = [B*T, N*H]
  key_layer <- tensorflow::tf$layers$dense(
    to_tensor_2d,
    num_attention_heads*size_per_head,
    activation = key_act,
    name = "key",
    kernel_initializer = create_initializer(initializer_range)
  )

  # `value_layer` = [B*T, N*H]
  value_layer <- tensorflow::tf$layers$dense(
    to_tensor_2d,
    num_attention_heads*size_per_head,
    activation = value_act,
    name = "value",
    kernel_initializer = create_initializer(initializer_range)
  )

  # `query_layer` = [B, N, F, H]
  query_layer <- transpose_for_scores(query_layer,
                                      batch_size,
                                      num_attention_heads,
                                      from_seq_length,
                                      size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer <- transpose_for_scores(key_layer,
                                    batch_size,
                                    num_attention_heads,
                                    to_seq_length,
                                    size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores <- tensorflow::tf$matmul(query_layer,
                                            key_layer,
                                            transpose_b = TRUE)
  attention_scores  <- tensorflow::tf$multiply(attention_scores,
                                               1.0/sqrt(size_per_head))

  if (!is.null(attention_mask)) {
    # `attention_mask` = [B, 1, F, T]
    # The axis argument is zero-indexed, so the expanded dimension is the
    # *second*, not the first.
    attention_mask <- tensorflow::tf$expand_dims(attention_mask,
                                                 axis = list(1L))

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder <- (1.0 - tensorflow::tf$cast(attention_mask,
                                        tensorflow::tf$float32))*(-10000.0)

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores <- attention_scores + adder
  }

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs <-  tensorflow::tf$nn$softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs <- dropout(attention_probs, attention_probs_dropout_prob)

  # The transpose_for_scores function wasn't used here in original Python code.
  # (The transformations were done explicitly instead.)
  # Not sure why, because it seems that this is what it was written for. -JDB

  # `value_layer` = [B, N, T, H]
  value_layer <- transpose_for_scores(input_tensor = value_layer,
                                      batch_size = batch_size,
                                      num_attention_heads = num_attention_heads,
                                      seq_length = to_seq_length,
                                      width = size_per_head)

  # `context_layer` = [B, N, F, H]
  context_layer <- tensorflow::tf$matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer <- tensorflow::tf$transpose(context_layer,
                                            perm = list(0L, 2L, 1L, 3L))

  if (do_return_2d_tensor) {
    context_layer <- tensorflow::tf$reshape(
      context_layer,
      list(batch_size*from_seq_length,
           as.integer(num_attention_heads*size_per_head)))
  } else {
    context_layer <- tensorflow::tf$reshape(
      context_layer,
      list(batch_size,
           as.integer(from_seq_length),
           as.integer(num_attention_heads),
           as.integer(size_per_head)))
  }
  # ATTN: modified below to include attention_data in return
  to_return <- list("context_layer" = context_layer,
                    "attention_data" = attention_probs)
  return(to_return)
  # return(context_layer)
  # ATTN: modified above to include attention_data in return
}

# transformer_model -------------------------------------------------------

#' Build multi-head, multi-layer Transformer
#'
#' Multi-headed, multi-layer Transformer from "Attention is All You Need". This
#' is almost an exact implementation of the original Transformer encoder.
#'
#' See the original paper: \url{https://arxiv.org/abs/1706.03762}
#'
#' Also see:
#' \url{https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py}
#'
#' @param input_tensor Float Tensor of shape \code{[batch_size, seq_length,
#'   hidden_size]}.
#' @param attention_mask (Optional) Integer Tensor of shape \code{batch_size,
#'   seq_length, seq_length}, with 1 for positions that can be attended to and 0
#'   in positions that should not be.
#' @param hidden_size Integer; hidden size of the Transformer.
#' @param num_hidden_layers Integer; number of layers (blocks) in the
#'   Transformer.
#' @param num_attention_heads Integer; number of attention heads in the
#'   Transformer.
#' @param intermediate_size Integer; the size of the "intermediate" (a.k.a.,
#'   feed forward) layer.
#' @param intermediate_act_fn The non-linear activation function to apply to the
#'   output of the intermediate/feed-forward layer. (Function, not character.)
#' @param hidden_dropout_prob Numeric; the dropout probability for the hidden
#'   layers.
#' @param attention_probs_dropout_prob Numeric; the dropout probability of the
#'   attention probabilities.
#' @param initializer_range Numeric; the range of the initializer (stddev of
#'   truncated normal).
#' @param do_return_all_layers Logical; whether to also return all layers or
#'   just the final layer. If this is TRUE, will also return attention
#'   probabilities.
#'
#' @return float Tensor of shape \code{[batch_size, seq_length, hidden_size]},
#'   the final hidden layer of the Transformer. Or if `do_return_all_layers` is
#'   `TRUE`, a list of such Tensors (one for each hidden layer).
#' @export
#'
#' @examples
#' \dontrun{
#' batch_size <- 10
#' seq_length <- 500
#' hidden_size <- 120
#'
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      {
#'        input_tensor <- tensorflow::tf$get_variable("input",
#'                                                    shape = c(batch_size,
#'                                                              seq_length,
#'                                                              hidden_size))
#'      }
#' )
#'
#' model_t <- transformer_model(input_tensor = input_tensor,
#'                              hidden_size = hidden_size)
#' }
transformer_model <- function(input_tensor,
                              attention_mask = NULL,
                              hidden_size = 768L,
                              num_hidden_layers = 12L,
                              num_attention_heads = 12L,
                              intermediate_size = 3072L,
                              intermediate_act_fn = gelu,
                              hidden_dropout_prob = 0.1,
                              attention_probs_dropout_prob = 0.1,
                              initializer_range = 0.02,
                              do_return_all_layers = FALSE) {
  if (hidden_size %% num_attention_heads != 0) {
    stop(paste("The hidden size:",
               hidden_size,
               "is not a multiple of the number of attention heads:",
               num_attention_heads))
  }

  attention_head_size <- hidden_size %/% num_attention_heads
  input_shape <- get_shape_list(input_tensor, expected_rank = 3L)
  batch_size <- input_shape[[1]]
  seq_length <- input_shape[[2]]
  input_width <- input_shape[[3]]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if (input_width != hidden_size) {
    stop(paste("The width of the input tensor:",
               input_width,
               "is not equal to the hidden size:",
               hidden_size))
  }

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output <- reshape_to_matrix(input_tensor)

  # ATTN: modified below to initialize all_attn_data list
  all_attn_data <- vector(mode = "list", length = num_hidden_layers)
  # ATTN: modified above to initialize all_attn_data list

  all_layer_outputs <- vector(mode = "list", length = num_hidden_layers)
  # probably want to refactor this loop later. For now, following python...
  for (layer_idx in 1:num_hidden_layers) { #... but starting at 1 rather than 0.
    # !! To correctly load the checkpoint parameters, need to follow python
    # names exactly. !!
    python_index <- layer_idx - 1
    scope_name <- paste0("layer_", python_index)
    with(tensorflow::tf$variable_scope(scope_name), {
      layer_input <- prev_output
      with(tensorflow::tf$variable_scope("attention"), {
        # The original python code here was apparently structured for
        # future expansion, making it easy(ish) to modify if
        # other sequences were added in parallel to the attention layers. (?)
        # I'm rewriting to make more sense as-is.
        # -JDB
        with(tensorflow::tf$variable_scope("self"), {
          attention_output <- attention_layer(
            from_tensor = layer_input,
            to_tensor = layer_input,
            attention_mask = attention_mask,
            num_attention_heads = num_attention_heads,
            size_per_head = attention_head_size,
            attention_probs_dropout_prob = attention_probs_dropout_prob,
            initializer_range = initializer_range,
            do_return_2d_tensor = TRUE,
            batch_size = batch_size,
            from_seq_length = seq_length,
            to_seq_length = seq_length)
        })
        # ATTN: modified below to separate out attention_data
        attention_data <- attention_output$attention_data
        attention_output <- attention_output$context_layer
        # ATTN: modified above to separate out attention_data

        with(tensorflow::tf$variable_scope("output"), {
          attention_output <- tensorflow::tf$layers$dense(
            attention_output,
            hidden_size,
            kernel_initializer = create_initializer(initializer_range))
          attention_output <- dropout(attention_output, hidden_dropout_prob)
          attention_output <- layer_norm(attention_output + layer_input)
        })
      })
      # The activation is only applied to the "intermediate" hidden layer.
      with(tensorflow::tf$variable_scope("intermediate"), {
        intermediate_output <- tensorflow::tf$layers$dense(
          attention_output,
          intermediate_size,
          activation = intermediate_act_fn,
          kernel_initializer = create_initializer(initializer_range))
      })
      # Down-project back to `hidden_size` then add the residual.
      with(tensorflow::tf$variable_scope("output"), {
        layer_output <- tensorflow::tf$layers$dense(
          intermediate_output,
          hidden_size,
          kernel_initializer = create_initializer(initializer_range))
        layer_output <- dropout(layer_output, hidden_dropout_prob)
        layer_output <- layer_norm(layer_output + attention_output)
        prev_output <- layer_output
        all_layer_outputs[[layer_idx]] <- layer_output

        # ATTN: modified below to store attention_data in all_attn_data list
        all_attn_data[[layer_idx]] <- attention_data
        # ATTN: modified above to store attention_data in all_attn_data list
      })
    })
  }
  if (do_return_all_layers) {
    final_outputs <- purrr::map(all_layer_outputs,
                                reshape_from_matrix,
                                input_shape)
    # ATTN: modified below to include all_attn_data in return
    to_return <- list("final_outputs" = final_outputs,
                      "attention_data" = all_attn_data)
    return(to_return)
    # return(final_outputs)
    # ATTN: modified above to include all_attn_data in return
  } else {
    final_output <- reshape_from_matrix(prev_output, input_shape)
    return(final_output)
  }
}

# get_shape_list ----------------------------------------------------------

#' Return the shape of tensor
#'
#' Returns a list of the shape of tensor, preferring static dimensions. (A
#' static dimension is known at graph definition time, and a dynamic dimension
#' is known only at graph execution time.)
#' https://stackoverflow.com/questions/37096225/
#'
#' @param tensor A tf.Tensor object to find the shape of.
#' @param expected_rank The expected rank of \code{tensor}, as an integer vector
#'   or list. If this is specified and the \code{tensor} has a rank not listed
#'   in \code{expected_rank}, an exception will be thrown.
#'
#' @param name Optional name of the tensor for the error message.
#'
#' @return A list of dimensions of the shape of tensor. All static dimensions
#'   will be returned as native integers, and dynamic dimensions will be
#'   returned as tf.Tensor scalars. (I'm not very comfortable with this
#'   behavior. It's not usually good practice to make the return type vary
#'   depending on the input.)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'  {
#'    phx <- tensorflow::tf$placeholder(tensorflow::tf$int32, shape = c(4))
#'    get_shape_list(phx) # static
#'    tfu <- tensorflow::tf$unique(phx)
#'    tfy <- tfu$y
#'    get_shape_list(tfy) # dynamic
#'  }
#' )
#' }
get_shape_list <- function(tensor, expected_rank = NULL, name = NULL) {
  if (is.null(name)) {
    name <- tensor$name
  }

  if (!is.null(expected_rank)) {
    assert_rank(tensor, expected_rank, name)
  }

  shape <- tensor$shape$as_list()

  shape <- as.list(shape) # for consistency

  # dynamic dimensions will be NULL in the shape vector.
  # When NULLs are list or vector elements it gets ... tricky in R.
  # I believe the following works, but there is likely a better way to do this.

  non_static_indexes <- c()

  for (index in seq_along(shape) ) {
    dim <- shape[index] # will now be "NULL" if dynamic dimension.
    if (dim == "NULL") {
      non_static_indexes <- c(non_static_indexes, index)
    }
  }

  if (length(non_static_indexes) == 0) {
    return(shape)
  }

  dyn_shape <- tensorflow::tf$shape(tensor)

  for (index in non_static_indexes) {
    # Note: the R tensorflow package now uses 1-based indexing by default.
    # ... but generally sticks to zero-indexing for Tensor indices.
    shape[[index]] <- dyn_shape[index]
  }

  return(shape)
}

# reshape_to_matrix ----------------------------------------------------------

#' Turn a tensor into a matrix
#'
#' Reshapes a >= rank 2 tensor to a rank 2 tensor. The last dimension is
#' preserved; the rest are flattened.
#'
#' @param input_tensor Tensor to reshape.
#'
#' @return The Tensor reshaped to rank 2.
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      r3t <- tensorflow::tf$get_variable("r3t", dtype = "int32",
#'                                         shape = c(10, 20, 3))
#' )
#' reshape_to_matrix(r3t)
#' }
reshape_to_matrix <- function(input_tensor) {
  ndims <- input_tensor$shape$ndims
  if (ndims == 2) {
    return(input_tensor)
  }

  if (ndims < 2) {
    stop(paste("Input tensor must have at least rank 2. Shape =",
               input_tensor$shape))
  }

  input_shape <- input_tensor$shape$as_list()
  width <- input_shape[[ndims]]
  output_tensor <- tensorflow::tf$reshape(input_tensor, list(-1L, width))
  return(output_tensor)
}

# reshape_from_matrix ----------------------------------------------------------

#' Turn a matrix into a tensor
#'
#' Reshapes a rank 2 tensor back to its original rank >= 2 tensor. The final
#' dimension ('width') of the tensor is assumed to be preserved. If a different
#' width is requested, function will complain.
#'
#' @param output_tensor Tensor to reshape. What a lousy name for an input.
#' @param orig_shape_list Shape to cast Tensor into.
#'
#' @return The Tensor reshaped to rank specified by orig_shape_list.
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                                    reuse = tensorflow::tf$AUTO_REUSE),
#'      r2t <- tensorflow::tf$get_variable("r2t", dtype = "int32",
#'                                         shape = c(10, 20))
#' )
#' reshape_from_matrix(r2t, orig_shape_list=c(5L, 2L, 20L))
#' }
reshape_from_matrix <- function(output_tensor, orig_shape_list) {
  output_shape <- get_shape_list(output_tensor)
  num_dims <- length(orig_shape_list)
  width <- orig_shape_list[num_dims]
  orig_dims <- orig_shape_list[-num_dims]

  # the following checks aren't in the python code, but seem natural to include
  if (length(output_shape) != 2) {
    stop("tensor is not rank 2")
  }
  if (output_shape[[2]] != width) {
    stop("width is not consistent")
  }

  if (length(orig_shape_list) == 2) {
    return(output_tensor)
  }
  return(tensorflow::tf$reshape(output_tensor,
                                orig_shape_list))
}

# assert_rank ----------------------------------------------------------

#' Confirm the rank of a tensor
#'
#' Throws an error if the tensor rank is not of the expected rank.
#'
#' @param tensor A tf.Tensor to check the rank of.
#' @param expected_rank  Integer vector or list of integers, expected rank.
#' @param name Optional name of the tensor for the error message.
#'
#' @return TRUE if the Tensor is of the expected rank (error otherwise).
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'                        reuse = tensorflow::tf$AUTO_REUSE),
#'      {
#'        ids <- tensorflow::tf$get_variable("x", dtype = "int32", shape = 10L)
#'        assert_rank(ids, 1)
#'        assert_rank(ids, 1:2)
#'        assert_rank(ids, 2)
#'      }
#' )
#' }
assert_rank <- function(tensor, expected_rank, name = NULL) {
  if (is.null(name)) {
    name <- tensor$name
  }

  actual_rank <- tensor$shape$ndims

  if (! actual_rank %in% expected_rank) {
    stop(paste0(
      "For the tensor ", name,
      ", the actual rank ", actual_rank,
      " (shape = ", tensor$shape,
      ") is not equal to the expected rank ", expected_rank, "."
    ))
  }
  return(TRUE)
}


