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


# class InputExample_EF ---------------------------------------------------

#' Construct objects of class \code{InputExample_EF}
#'
#' An InputExample_EF is a single test example for feature extraction. Note that
#' this class is similiar to the InputExample class used for simple sequence
#' classification, but doesn't have a label property. The name of the id
#' property is also annoyingly different; should eventually standardize better
#' than the Python folks did. (RBERT issue #28.)
#'
#' @param unique_id Integer or character; a unique id for this example.
#' @param text_a Character; the untokenized text of the first sequence.
#' @param text_b (Optional) Character; the untokenized text of the second
#'   sequence.
#'
#' @return An object of class \code{InputExample_EF}.
#' @export
#'
#' @examples
#' input_ex <- InputExample_EF(
#'   unique_id = 1,
#'   text_a = "I work at the bank."
#' )
InputExample_EF <- function(unique_id,
                            text_a,
                            text_b = NULL) {
  obj <- list(
    "unique_id" = unique_id,
    "text_a" = text_a,
    "text_b" = text_b
  )
  class(obj) <- "InputExample_EF"
  return(obj)
}
# class InputFeatures_EF --------------------------------------------------

#' Construct objects of class \code{InputFeatures_FE}
#'
#' An InputFeatures object is a single set of (input) features of data used for
#' (output) feature extraction. Note that this class is similiar to the
#' InputFeatures class used for simple sequence classification, with annoying
#' differences. Will eventually standardize; till then, check parameter names.
#' (RBERT issue #28.)
#'
#' @param unique_id Integer or character; a unique id for this example.
#' @param tokens Character vector; the actual tokens in this example.
#' @param input_ids Integer vector; the sequence of token ids in this example.
#' @param input_mask Integer vector; sequence of 1s (for "real" tokens) and 0s
#'   (for padding tokens).
#' @param input_type_ids Integer vector; aka token_type_ids. Indicators for
#'   which sentence (or sequence) each token belongs to. Classical BERT supports
#'   only 0s and 1s (for first and second sentence, respectively).
#'
#' @return An object of class \code{InputFeatures_FE}.
#' @keywords internal
.InputFeatures_EF <- function(unique_id,
                              tokens,
                              input_ids,
                              input_mask,
                              input_type_ids) {
  obj <- list(
    "unique_id" = unique_id,
    "tokens" = tokens,
    "input_ids" = input_ids,
    "input_mask" = input_mask,
    "input_type_ids" = input_type_ids
  )
  class(obj) <- "InputFeatures"
  return(obj)
}

# input_fn_builder_EF -----------------------------------------------------


#' Create an \code{input_fn} closure to be passed to TPUEstimator
#'
#' Creates an \code{input_fn} closure to be passed to TPUEstimator. The output
#' of this closure is the (modified) output of
#' \code{tensorflow::tf$data$Dataset$from_tensor_slices} (an object of class
#' "tensorflow.python.data.ops.dataset_ops.BatchDataset"). This function is
#' similar to \code{input_fn_builder} from run_classifier.R. (RBERT issue #28.)
#'
#' @param features A list of features (objects of class
#'   \code{InputFeatures_EF}).
#' @param seq_length Integer; the maximum length (number of tokens) of each
#'   example. (Examples should already be padded to this length by this point.)
#'
#' @return An \code{input_fn} closure to be passed to TPUEstimator.
#' @keywords internal
input_fn_builder_EF <- function(features,
                                seq_length) {
  all_unique_ids <- purrr::map(
    features,
    function(f) {
      as.integer(f$unique_id)
    }
  )
  all_input_ids <- purrr::map(
    features,
    function(f) {
      as.integer(f$input_ids)
    }
  )
  all_input_mask <- purrr::map(
    features,
    function(f) {
      as.integer(f$input_mask)
    }
  )
  all_input_type_ids <- purrr::map(
    features,
    function(f) {
      as.integer(f$input_type_ids)
    }
  )

  input_fn <- function(params) {
    batch_size <- params$batch_size

    num_examples <- length(features)

    # "This is for demo purposes and does NOT scale to large data sets. We do
    #  not use Dataset.from_generator() because that uses tf.py_func which is
    #  not TPU compatible. The right way to load data is with TFRecordReader."

    d <- tensorflow::tf$data$Dataset$from_tensor_slices(
      # "A nested structure of tensors, each having the same size in the 0th
      # dimension." Try as a list. -JDB
      list(
        "unique_ids" = tensorflow::tf$constant(
          all_unique_ids,
          shape = tensorflow::shape(num_examples),
          dtype = tensorflow::tf$int32
        ),
        "input_ids" = tensorflow::tf$constant(
          all_input_ids,
          shape = tensorflow::shape(num_examples, seq_length),
          dtype = tensorflow::tf$int32
        ),
        "input_mask" = tensorflow::tf$constant(
          all_input_mask,
          shape = tensorflow::shape(num_examples, seq_length),
          dtype = tensorflow::tf$int32
        ),
        "input_type_ids" = tensorflow::tf$constant(
          all_input_type_ids,
          shape = tensorflow::shape(num_examples, seq_length),
          dtype = tensorflow::tf$int32
        )
      )
    )

    d <- d$batch(
      batch_size = batch_size,
      drop_remainder = FALSE
    )
    return(d) # return from `input_fn`
  }
  return(input_fn)
}
# .model_fn_builder_EF -----------------------------------------------------

#' Define \code{model_fn} closure for \code{TPUEstimator}
#'
#' Returns \code{model_fn} closure, which is an input to \code{TPUEstimator}.
#' This function is similar to \code{model_fn_builder} from run_classifier.R.
#' (RBERT issue #28.)
#'
#' The \code{model_fn} function takes four parameters: \describe{
#' \item{features}{A list (or similar structure) that contains objects such as
#' \code{input_ids}, \code{input_mask}, \code{tokens},  and
#' \code{input_type_ids}. These objects will be inputs to the
#' \code{create_model} function.} \item{labels}{Not used in this function, but
#' presumably we need to keep this slot here.} \item{mode}{Character; value such
#' as "train", "infer", or "eval".} \item{params}{Not used in this function, but
#' presumably we need to keep this slot here.} }
#'
#' The output of \code{model_fn} is the result of a
#' \code{tf$contrib$tpu$TPUEstimatorSpec} call.
#'
#' @param bert_config \code{BertConfig} instance.
#' @param init_checkpoint Character; path to the checkpoint directory, plus
#'   checkpoint name stub (e.g. "bert_model.ckpt"). Path must be absolute and
#'   explicit, starting with "/".
#' @param layer_indexes Integer list; indexes (positive, or negative counting
#'   back from the end) indicating which layers to extract as "output features".
#'   (It needs to be specified here because we get them back as the model
#'   "predictions".)
#' @param use_tpu Logical; whether to use TPU.
#' @param use_one_hot_embeddings Logical; whether to use one-hot word embeddings
#'   or tf.embedding_lookup() for the word embeddings.
#'
#' @return \code{model_fn} closure for \code{TPUEstimator}.
#' @keywords internal
.model_fn_builder_EF <- function(bert_config,
                                 init_checkpoint,
                                 layer_indexes,
                                 use_tpu,
                                 use_one_hot_embeddings) {
  # The `model_fn` for TPUEstimator.
  model_fn <- function(features, labels, mode, params) {
    unique_ids <- features$unique_ids
    input_ids <- features$input_ids
    input_mask <- features$input_mask
    input_type_ids <- features$input_type_ids
    input_shape <- get_shape_list(input_ids, expected_rank = 2L)

    model <- BertModel(
      config = bert_config,
      is_training = FALSE,
      input_ids = input_ids,
      input_mask = input_mask,
      token_type_ids = input_type_ids,
      use_one_hot_embeddings = use_one_hot_embeddings
    )

    if (mode != tensorflow::tf$estimator$ModeKeys$PREDICT) {
      stop("Only PREDICT modes are supported.") # nocov
    }

    tvars <- tensorflow::tf$trainable_variables()
    initialized_variable_names <- list()
    scaffold_fn <- NULL

    gamap <- get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    assignment_map <- gamap$assignment_map
    initialized_variable_names <- gamap$initialized_variable_names
    if (use_tpu) { # nocov start
      tpu_scaffold <- function() {
        tensorflow::tf$train$init_from_checkpoint(
          init_checkpoint,
          assignment_map
        )
        return(tensorflow::tf$train$Scaffold())
      }
      scaffold_fn <- tpu_scaffold # nocov end
    } else {
      tensorflow::tf$train$init_from_checkpoint(
        init_checkpoint,
        assignment_map
      )
    }

    all_layers <- model$all_encoder_layers

    # ATTN: modified below to get attention_data from model
    attention_data <- model$attention_data
    # ATTN: modified above to get attention_data from model

    predictions <- list()
    predictions[["unique_id"]] <- unique_ids

    # Always include raw embeddings as the zeroth layer "output". We'll filter
    # them back out if we don't want them.
    predictions[["layer_output_0"]] <- model$embedding_output

    for (i in seq_along(layer_indexes)) {
      layer_index <- layer_indexes[[i]]
      # Accomodate both positive and negative indices.
      # Note that `all_layers` is 1-indexed!
      actual_index <- .get_actual_index(layer_index, length(all_layers))
      # For clarity, always use actual index to label outputs.
      key_str <- paste0("layer_output_", actual_index)
      predictions[[key_str]] <- all_layers[[actual_index]]

      # ATTN: modified below to include attention_data in "predictions"
      attn_key_str <- paste0("layer_attention_", actual_index)
      predictions[[attn_key_str]] <- attention_data[[actual_index]]
      # ATTN: modified above to include attention_data in "predictions"
    }
    output_spec <- tensorflow::tf$contrib$tpu$TPUEstimatorSpec(
      mode = mode,
      predictions = predictions,
      scaffold_fn = scaffold_fn
    )
    return(output_spec)
  } # end of `model_fn` definition
  return(model_fn)
}



# .convert_single_example_EF -----------------------------------------------

#' Convert a single \code{InputExample_EF} into a single \code{InputFeatures_EF}
#'
#' Converts a single \code{InputExample_EF} into a single
#' \code{InputFeatures_EF}. Very similar to \code{convert_single_example} from
#' run_classifier.R. (RBERT issue #28.)
#'
#' @param ex_index Integer; the index of this example. This is used to determine
#'   whether or not to print out some log info (for debugging or runtime
#'   confirmation). It is assumed this starts with 1 (in R).
#' @param example The \code{InputExample_EF} to convert.
#' @param seq_length Integer; the maximum number of tokens that will be
#'   considered together.
#' @param tokenizer A tokenizer object to use (e.g. object of class
#'   FullTokenizer).
#'
#' @return An object of class \code{InputFeatures_EF}.
#' @keywords internal
.convert_single_example_EF <- function(ex_index,
                                       example,
                                       seq_length,
                                       tokenizer) {

  # note use of S3 classes for dispatch, not methods.
  tokens_a <- tokenize(tokenizer, example$text_a)
  tokens_b <- NULL
  if (!is.null(example$text_b)) {
    tokens_b <- tokenize(tokenizer, example$text_b)
  }

  if (!is.null(tokens_b)) {
    # Modifies `tokens_a` and `tokens_b` so that the total length is less than
    # the specified length. Account for [CLS], [SEP], [SEP] with "- 3"
    truncated_seq <- truncate_seq_pair(
      tokens_a, tokens_b,
      seq_length - 3
    )
    tokens_a <- truncated_seq$trunc_a
    tokens_b <- truncated_seq$trunc_b
  } else {
    # Account for [CLS] and [SEP] with "- 2"
    if (length(tokens_a) > seq_length - 2) {
      tokens_a <- tokens_a[1:(seq_length - 2)]
    }
  }

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.

  # The next few lines of code just insert the "[CLS]" and "[SEP]" tokens
  # in the appropriate places, and create the type_ids list.  -JDB
  cls_token <- "[CLS]"
  sep_token <- "[SEP]"
  tokens <- unlist(list(cls_token, tokens_a, sep_token))
  input_type_ids <- rep(0, length(tokens))

  if (!is.null(tokens_b)) {
    tokens2 <- unlist(list(tokens_b, sep_token))
    input_type_ids2 <- rep(1, length(tokens2))
    tokens <- c(tokens, tokens2)
    input_type_ids <- c(input_type_ids, input_type_ids2)
  }
  input_ids <- convert_tokens_to_ids(tokenizer$vocab, tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask <- rep(1, length(input_ids))

  # Zero-pad up to the sequence length.
  pad_length <- seq_length - length(input_ids)
  padding <- rep(0, pad_length)
  input_ids <- c(input_ids, padding)
  input_mask <- c(input_mask, padding)
  input_type_ids <- c(input_type_ids, padding)

  # Stop now if the lengths aren't right somehow. -JDB
  if (length(input_ids) != seq_length |
    length(input_mask) != seq_length |
    length(input_type_ids) != seq_length) {
    stop("input_ids, input_mask, or input_type_ids are wrong length.") # nocov
  }

  feature <- .InputFeatures_EF(
    unique_id = example$unique_id,
    tokens = tokens,
    input_ids = input_ids,
    input_mask = input_mask,
    input_type_ids = input_type_ids
  )

  return(feature)
}

# .convert_examples_to_features_EF -----------------------------------------

#' Convert \code{InputExample_EF}s to \code{InputFeatures_EF}
#'
#' Converts a set of \code{InputExample_EF}s to a list of
#' \code{InputFeatures_EF}. Very similar to \code{convert_examples_to_features}
#' from run_classifier.R. (RBERT issue #28.)
#'
#' @param examples List of \code{InputExample_EF}s to convert.
#' @param seq_length Integer; the maximum number of tokens that will be
#'   considered together.
#' @param tokenizer A tokenizer object to use (e.g. object of class
#'   FullTokenizer).
#'
#' @return A list of \code{InputFeatures}.
#' @keywords internal
.convert_examples_to_features_EF <- function(examples,
                                             seq_length,
                                             tokenizer) {
  # I have no idea why they had to rename the elements of examples/features
  # and then recreate these functions to handle the slightly different versions.
  # Whatever. We will clean up later. -JDB
  example_indices <- seq_along(examples)
  num_examples <- length(examples)
  features <- purrr::map2(
    example_indices,
    examples,
    function(ex_index, example,
             seq_length, tokenizer) {
      .convert_single_example_EF(
        ex_index = ex_index,
        example = example,
        seq_length = seq_length,
        tokenizer = tokenizer
      )
    },
    seq_length, tokenizer
  )
  return(features)
}



# extract_features --------------------------------------------------------

#' Extract output features from BERT
#'
#' Given example sentences (as a list of \code{InputExample_EF}s), apply an
#' existing BERT model and capture certain output layers. (These could
#' potentially be used as features in downstream tasks.)
#'
#' @param examples List of \code{InputExample_EF}s to convert.
#' @param ckpt_dir Character; path to checkpoint directory. If specified, any
#'   other checkpoint files required by this function (\code{vocab_file},
#'   \code{bert_config_file}, or \code{init_checkpoint}) will default to
#'   standard filenames within \code{ckpt_dir}.
#' @param vocab_file path to vocabulary file. File is assumed to be a text file,
#'   with one token per line, with the line number corresponding to the index of
#'   that token in the vocabulary.
#' @param bert_config_file Character; the path to a json config file.
#' @param init_checkpoint Character; path to the checkpoint directory, plus
#'   checkpoint name stub (e.g. "bert_model.ckpt"). Path must be absolute and
#'   explicit, starting with "/".
#' @param output_file (optional) Character; file path (stub) for writing output
#'   to.
#' @param max_seq_length Integer; the maximum number of tokens that will be
#'   considered together.
#' @param layer_indexes Integer vector; indexes (positive, or negative counting
#'   back from the end) indicating which layers to extract as "output features".
#'   The "zeroth" layer embeddings are the input embeddings vectors to the first
#'   layer.
#' @param use_one_hot_embeddings Logical; whether to use one-hot word embeddings
#'   or tf.embedding_lookup() for the word embeddings.
#' @param batch_size Integer; how many examples to process per batch.
#' @param features Character; whether to return "output" (layer outputs, the
#'   default), "attention" (attention probabilities), or both.
#'
#' @return A list with elements "output" (the layer outputs as a tibble) and/or
#'   "attention" (the attention weights as a tibble).
#' @export
#'
#' @examples
#' \dontrun{
#' BERT_PRETRAINED_DIR <- download_BERT_checkpoint("bert_base_uncased")
#' vocab_file <- file.path(BERT_PRETRAINED_DIR, "vocab.txt")
#' init_checkpoint <- file.path(BERT_PRETRAINED_DIR, "bert_model.ckpt")
#' bert_config_file <- file.path(BERT_PRETRAINED_DIR, "bert_config.json")
#' examples <- list(
#'   InputExample_EF(
#'     unique_id = 1,
#'     text_a = "I saw the branch on the bank."
#'   ),
#'   InputExample_EF(
#'     unique_id = 2,
#'     text_a = "I saw the branch of the bank."
#'   )
#' )
#' feats <- extract_features(
#'   examples = examples,
#'   vocab_file = vocab_file,
#'   bert_config_file = bert_config_file,
#'   init_checkpoint = init_checkpoint,
#'   batch_size = 2L
#' )
#' # can just specify checkpoint directory
#' feats <- extract_features(
#'   examples = examples,
#'   ckpt_dir = BERT_PRETRAINED_DIR,
#'   batch_size = 2L
#' )
#' }
extract_features <- function(examples,
                             model = c(
                               "bert_base_uncased",
                               "bert_base_cased",
                               "bert_large_uncased",
                               "bert_large_cased",
                               "bert_large_uncased_wwm",
                               "bert_large_cased_wwm",
                               "bert_base_multilingual_cased",
                               "bert_base_chinese",
                               "scibert_scivocab_uncased",
                               "scibert_scivocab_cased",
                               "scibert_basevocab_uncased",
                               "scibert_basevocab_cased"
                             ),
                             ckpt_dir = NULL,
                             vocab_file = find_vocab(ckpt_dir),
                             bert_config_file = find_config(ckpt_dir),
                             init_checkpoint = find_ckpt(ckpt_dir),
                             output_file = NULL,
                             max_seq_length = 128L,
                             layer_indexes = -4:-1,
                             use_one_hot_embeddings = FALSE,
                             batch_size = 2L,
                             features = c(
                               "output",
                               "attention"
                             )) {
  # If they specify a ckpt_dir or all of the other files, use that info.
  # Otherwise infer from model.
  if ((missing(vocab_file) |
    missing(bert_config_file) |
    missing(init_checkpoint)) &
    is.null(ckpt_dir)) {
    if (missing(model)) {
      stop(
        "You must specify a model, a ckpt_dir, or the locations of ",
        "vocab_file, bert_config_file, and init_checkpoint."
      )
    } else {
      model <- match.arg(model)
      dir <- .choose_BERT_dir(NULL)
      ckpt_dir <- .get_model_subdir(model, dir)
      has_checkpoint <- .has_checkpoint(
        model = model,
        dir = dir,
        ckpt_dir = ckpt_dir
      )
      if (!has_checkpoint) {
        if (interactive()) { # nocov start
          do_download <- utils::menu(
            c("Yes.", "No."),
            title = paste(
              "Model not found. Do you wish to download the model?",
              "\nThis may take a long time and use a lot of disk space."
            )
          )
          if (do_download == 1L) {
            download_BERT_checkpoint(model, dir)
          } else {
            stop(
              "Could not find the specified model. Specify ckpt_dir, or ",
              "run download_BERT_checkpoint."
            )
          }
          # nocov end
        } else {
          stop(
            "Could not find the specified model. Specify ckpt_dir, or ",
            "run download_BERT_checkpoint."
          )
        }
      }

      # If we made it here, they have the model, so set the file locations.
      vocab_file <- find_vocab(ckpt_dir)
      bert_config_file <- find_config(ckpt_dir)
      init_checkpoint <- find_ckpt(ckpt_dir)
    }
  }

  if (missing(features)) {
    features <- "output"
  }
  features <- match.arg(features, several.ok = TRUE)

  include_zeroth <- FALSE
  if (0 %in% layer_indexes) {
    include_zeroth <- TRUE
    layer_indexes <- layer_indexes[layer_indexes != 0]
  }

  bert_config <- bert_config_from_json_file(bert_config_file)
  n_layers <- bert_config$num_hidden_layers
  layer_indexes_actual <- .get_actual_indexes(layer_indexes, n_layers)
  n_dimensions <- bert_config$hidden_size
  tokenizer <- FullTokenizer(
    vocab_file = vocab_file,
    do_lower_case = TRUE
  )
  is_per_host <- tensorflow::tf$contrib$tpu$InputPipelineConfig$PER_HOST_V2

  run_config <- tensorflow::tf$contrib$tpu$RunConfig(
    master = NULL, # assume for now *not* for TPU
    tpu_config = tensorflow::tf$contrib$tpu$TPUConfig(
      num_shards = 8L,
      per_host_input_for_training = is_per_host
    )
  )

  raw_features <- .convert_examples_to_features_EF(
    examples = examples,
    seq_length = max_seq_length,
    tokenizer = tokenizer
  )
  unique_id_to_feature <- list()
  for (feature in raw_features) {
    unique_id_to_feature[[feature$unique_id]] <- feature
  }

  model_fn <- .model_fn_builder_EF(
    bert_config = bert_config,
    init_checkpoint = init_checkpoint,
    layer_indexes = layer_indexes_actual,
    use_tpu = FALSE,
    use_one_hot_embeddings = use_one_hot_embeddings
  )

  estimator <- tensorflow::tf$contrib$tpu$TPUEstimator(
    use_tpu = FALSE, # no tpu support for now
    model_fn = reticulate::py_func(model_fn),
    config = run_config,
    predict_batch_size = batch_size
  )

  input_fn <- input_fn_builder_EF(
    features = raw_features,
    seq_length = max_seq_length
  )

  result_iterator <- estimator$predict(reticulate::py_func(input_fn),
    yield_single_examples = TRUE
  )


  # Set up the needed lists. They'll be filled in the for below.
  big_output <- NULL
  big_attention <- NULL
  wants_output <- "output" %in% features
  wants_attention <- "attention" %in% features
  if (wants_output) {
    big_output <- tibble::tibble(
      sequence_index = integer(),
      segment_index = integer(),
      token_index = integer(),
      token = character(),
      layer_index = integer()
    )
    for (colname in paste0("V", seq_len(n_dimensions))) {
      big_output[[colname]] <- integer()
    }
    layer_indexes_output <- layer_indexes_actual
    if (include_zeroth) {
      layer_indexes_output <- c(0L, layer_indexes_output)
    }
  }
  if (wants_attention) {
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
  }
  token_map <- tibble::tibble(
    sequence_index = integer(),
    token_index = integer(),
    segment_index = integer(),
    token = character()
  )

  # For speed testing.
  # wants_output <- FALSE
  # wants_attention <- FALSE

  # result_iterator should have an entry available for each sequence.
  for (i in seq_along(examples)) {
    result <- if ("next" %in% names(result_iterator)) {
      result_iterator$`next`() # nocov
    } else {
      result_iterator$`__next__`() # nocov
    }

    tokens <- unique_id_to_feature[[result$unique_id]]$tokens
    token_seq <- seq_along(tokens)
    token_map <- dplyr::bind_rows(
      token_map,
      data.frame(
        sequence_index = as.integer(result$unique_id),
        token_index = token_seq,
        token = tokens,
        stringsAsFactors = FALSE
      )
    )

    if (wants_output) {
      big_output <- .process_output_result(
        big_output,
        result,
        layer_indexes_output,
        token_seq
      )
    }

    if (wants_attention) {
      big_attention <- .process_attention_result(
        big_attention,
        result,
        layer_indexes_actual,
        token_seq
      )
    }
  }

  # Iterate one more time to let python finish and be happy.
  result <- tryCatch(
    {
      if ("next" %in% names(result_iterator)) {
        result_iterator$`next`() # nocov
      } else {
        result_iterator$`__next__`() # nocov
      }
    },
    error = function(e) {
      FALSE
      # If we get error, `result` will be assigned this FALSE.
      # The only way to tell we've reached the end is to get an error. :-/
    }
  )
  if (!identical(result, FALSE)) {
    stop("More results returned than sequences.") # nocov
  }

  # Clean up features.
  token_map <- .infer_segment_index(token_map, sequence_index)
  if (wants_output) {
    big_output <- .finalize_output(big_output, token_map)
  }
  if (wants_attention) {
    big_attention <- .finalize_attention(big_attention, token_map)
  }

  # I do it this way so, if they're NULL, that value won't appear in the list,
  # rather than appearing there as "NULL" like it would if I set this up in one
  # step.
  to_return <- list()
  to_return$output <- big_output
  to_return$attention <- big_attention
  return(to_return)
}


# .process_output_results -------------------------------------------------

.process_output_result <- function(big_output,
                                   result,
                                   layer_indexes_output,
                                   token_seq) {
  sequence_index <- as.integer(result$unique_id)
  result_output_names <- paste0("layer_output_", layer_indexes_output)
  sequence_outputs <- purrr::imap_dfr(
    result_output_names,
    function(this_layer, this_index) {
      this_output <- as.data.frame(
        result[[this_layer]][token_seq, ]
      )
      this_output[["layer_index"]] <- layer_indexes_output[[this_index]]
      this_output
    }
  )
  sequence_outputs[["token_index"]] <- rep(
    token_seq, length(result_output_names)
  )
  sequence_outputs[["sequence_index"]] <- sequence_index
  return(
    dplyr::bind_rows(
      big_output,
      sequence_outputs
    )
  )
}

.infer_segment_index <- function(feature_tibble, ...) {
  return(
    dplyr::ungroup(
      dplyr::select(
        dplyr::mutate(
          dplyr::group_by(
            dplyr::mutate(
              feature_tibble,
              is_sep = token == "[SEP]"
            ),
            ...
          ),
          segment_index = cumsum(is_sep) - is_sep + 1L
        ),
        -is_sep
      )
    )
  )
}

.finalize_output <- function(big_output, token_map) {
  return(
    dplyr::select(
      dplyr::mutate(
        dplyr::left_join(
          big_output, token_map,
          by = c("sequence_index", "token_index"),
          suffix = c("", "_fill")
        ),
        segment_index = segment_index_fill,
        token = token_fill
      ),
      -dplyr::ends_with("_fill")
    )
  )
}


# .process_attention_results ----------------------------------------------

.process_attention_result <- function(big_attention,
                                      result,
                                      layer_indexes_actual,
                                      token_seq) {
  result_attention_names <- paste0("layer_attention_", layer_indexes_actual)

  sequence_attention <- purrr::imap_dfr(
    result_attention_names,
    function(this_layer, this_index) {
      this_attention <- tidyr::unnest_longer(
        tidyr::unnest_longer(
          tibble::enframe(
            purrr::array_tree(
              result[[this_layer]][, token_seq, token_seq]
            ),
            name = "head_index"
          ),
          value,
          indices_to = "token_index"
        ),
        value,
        indices_to = "attention_token_index",
        values_to = "attention_weight"
      )
      this_attention$layer_index <- layer_indexes_actual[[this_index]]
      this_attention
    }
  )
  sequence_attention$sequence_index <- as.integer(result$unique_id)
  big_attention <- dplyr::bind_rows(
    big_attention,
    sequence_attention
  )

  return(big_attention)
}

.finalize_attention <- function(big_attention,
                                token_map) {
  return(
    dplyr::select(
      dplyr::mutate(
        dplyr::left_join(
          dplyr::select(
            dplyr::mutate(
              dplyr::left_join(
                big_attention, token_map,
                by = c("sequence_index", "token_index"),
                suffix = c("", "_fill")
              ),
              segment_index = segment_index_fill,
              token = token_fill
            ),
            -dplyr::ends_with("_fill")
          ),
          token_map,
          by = c("sequence_index", "attention_token_index" = "token_index"),
          suffix = c("", "_fill")
        ),
        attention_segment_index = segment_index_fill,
        attention_token = token_fill
      ),
      -dplyr::ends_with("_fill")
    )
  )
}


# .get_actual_index ---------------------------------------------------

#' Standardize Indices
#'
#' Convert negative indices to positive ones. Use the convention that
#' \code{vec[[-1L]]} signifies the last element of \code{vec}, \code{vec[[-2L]]}
#' signifies the second-to-last element of \code{vec}, and so on. 1-based
#' indexing is assumed. Values of zero, or out-of-range indices, will be
#' rejected.
#'
#' @param index Integer; the index to normalize.
#' @param length Integer; the length of the vector or list we are indexing.
#'
#' @return The "actual" integer index, between 1 and \code{length}, inclusive.
#' @keywords internal
.get_actual_index <- function(index,
                              length) {
  index <- as.integer(index)
  if (abs(index) > length) {
    stop(paste(
      "Index out of range.",
      "Absolute value of index must be within specified length."
    ))
  } else if (index == 0) {
    stop(paste(
      "Ambiguous index.",
      "Only strictly positive or negative indices accepted."
    ))
  } else if (index < 0) {
    return(as.integer((length + index) %% length + 1L))
  } else {
    return(index)
  }
}

.get_actual_indexes <- function(indexes, object_length) {
  return(
    purrr::map_int(indexes, .get_actual_index, object_length)
  )
}

# make_examples_simple ----------------------------------------------------

#' Easily make examples for BERT
#'
#' A simple wrapper function to turn a list of text (as a character
#' vector or list) into a list of examples suitable for use with RBERT. If the
#' input is a flat list or vector of characters, the examples will be
#' single-segment, with NULL for the second segment. If the input contains
#' length-2 sublists or vectors, those examples will be two-segment sequences,
#' e.g. for doing sentence-pair classification.
#'
#' @param seq_list Character vector or list; text to turn into examples.
#'
#' @return A list of \code{InputExample_EF} objects.
#' @export
#'
#' @examples
#' input_ex <- make_examples_simple(c(
#'   "Here are some words.",
#'   "Here are some more words."
#' ))
#' input_ex2 <- make_examples_simple(list(
#'   c(
#'     "First sequence, first segment.",
#'     "First sequence, second segment."
#'   ),
#'   c(
#'     "Second sequence, first segment.",
#'     "Second sequence, second segment."
#'   )
#' ))
make_examples_simple <- function(seq_list) {
  if (any(purrr::map_int(seq_list, length) > 2)) {
    warning(
      "Examples must contain at most two distinct segments. ",
      "Segments beyond the second will be ignored."
    )
  }
  seq_nums <- seq_along(seq_list)
  purrr::map(seq_nums, function(sn) {
    first_segment <- seq_list[[sn]][[1]]
    second_segment <- NULL
    if (length(seq_list[[sn]]) > 1) {
      second_segment <- seq_list[[sn]][[2]]
    }
    InputExample_EF(
      unique_id = sn,
      text_a = first_segment,
      text_b = second_segment
    )
  })
}
