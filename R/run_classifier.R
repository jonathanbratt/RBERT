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


# class InputExample ------------------------------------------------------------

#' Construct objects of class \code{InputExample}
#'
#' An input example is a single training/test example for simple sequence
#' classification.
#'
#' @param guid Unique id for the example (character or integer?).
#' @param text_a Character; the untokenized text of the first sequence. For
#'   single sequence tasks, only this sequence must be specified.
#' @param text_b (Optional) Character; the untokenized text of the second
#'   sequence. Only must be specified for sequence pair tasks.
#' @param label (Optional) Character; the label of the example. This should be
#'   specified for train and dev examples, but not for test examples.
#'
#' @return An object of class \code{InputExample}.
#' @export
#'
#' @examples
#' \dontrun{
#' input_ex <- InputExample(guid = 0, text_a = "Some text to classify.")
#' }
InputExample <- function(guid,
                         text_a,
                         text_b = NULL,
                         label = NULL) {
  obj <- list(
    "guid" = guid,
    "text_a" = text_a,
    "text_b" = text_b,
    "label" = label
  )
  class(obj) <- "InputExample"
  return(obj)
}

# class PaddingInputExample -----------------------------------------------------

# not sure we need this one.


# class InputFeatures -----------------------------------------------------------

#' Construct objects of class \code{InputFeatures}
#'
#' An InputFeatures object is a single set of features of data.
#'
#' @param input_ids Integer Tensor; the sequence of token ids in this example.
#' @param input_mask Integer Tensor; sequence of 1s (for "real" tokens) and 0s
#'   (for padding tokens).
#' @param segment_ids Integer Tensor; aka token_type_ids. Indicators for which
#'   sentence (or sequence each token belongs to). Classical BERT supports only
#'   0s and 1s (for first and second sentence, respectively).
#' @param label_id Integer; represents training example classification labels.
#' @param is_real_example Logical; later on this is used as a flag for whether
#'   to "count" this example for calculating accuracy and loss.
#'
#' @return An object of class \code{InputFeatures}.
#' @export
#'
#' @examples
#' \dontrun{
#' features <- InputFeatures(input_ids, input_mask, segment_ids, label_id)
#' }
InputFeatures <- function(input_ids,
                          input_mask,
                          segment_ids,
                          label_id,
                          is_real_example = TRUE) {
  obj <- list(
    "input_ids" = input_ids,
    "input_mask" = input_mask,
    "segment_ids" = segment_ids,
    "label_id" = label_id,
    "is_real_example" = is_real_example
  )
  class(obj) <- "InputFeatures"
  return(obj)
}



# convert_single_example --------------------------------------------------

#' Convert a single \code{InputExample} into a single \code{InputFeatures}
#'
#' Converts a single \code{InputExample} into a single \code{InputFeatures}.
#'
#' @param ex_index Integer; the index of this example. This is used to determine
#'   whether or not to print out some log info (for debugging or runtime
#'   confirmation). It is assumed this starts with 1 (in R).
#' @param example The \code{InputExample} to convert.
#' @param label_list Character (or integer); allowed labels for these examples.
#' @param max_seq_length Integer; the maximum number of tokens that will be
#'   considered together.
#' @param tokenizer A tokenizer object to use (e.g. object of class
#'   FullTokenizer).
#'
#' @return An object of class \code{InputFeatures}.
#' @export
#'
#' @examples
#' \dontrun{
#' tokenizer <- FullTokenizer("vocab.txt")
#' input_ex <- InputExample(
#'   guid = 1L,
#'   text_a = "Some text to classify.",
#'   text_b = "More wordy words.",
#'   label = "good"
#' )
#' feat <- convert_single_example(
#'   ex_index = 1L,
#'   example = input_ex,
#'   label_list = c("good", "bad"),
#'   max_seq_length = 15L,
#'   tokenizer = tokenizer
#' )
#' }
convert_single_example <- function(ex_index,
                                   example,
                                   label_list,
                                   max_seq_length,
                                   tokenizer) {
  # Use the same class name for padding examples... -JDB
  if (inherits(example, "PaddingInputExample")) {
    return(InputFeatures(
      input_ids = rep(0, max_seq_length),
      input_mask = rep(0, max_seq_length),
      segment_ids = rep(0, max_seq_length),
      label_id = 0,
      is_real_example = FALSE
    ))
  }

  # I'm going to tentatively use 1-based indexing here. -JDB
  label_map <- seq_along(label_list)
  names(label_map) <- label_list

  # note use of S3 classes for dispatch, not methods.
  tokens_a <- tokenize(tokenizer, example$text_a)
  tokens_b <- NULL
  if (!is.null(example$text_b)) {
    tokens_b <- tokenize(tokenizer, example$text_b)
  }

  if (!is.null(tokens_b)) {
    # Modifies `tokens_a` and `tokens_b` so that the total length is less than
    # the specified length. Account for [CLS], [SEP], [SEP] with "- 3"
    truncated_seq <- truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    tokens_a <- truncated_seq$trunc_a
    tokens_b <- truncated_seq$trunc_b
  } else {
    # Account for [CLS] and [SEP] with "- 2"
    if (length(tokens_a) > max_seq_length - 2) {
      tokens_a <- tokens_a[1:(max_seq_length - 2)]
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
  segment_ids <- rep(0, length(tokens))

  if (!is.null(tokens_b)) {
    tokens2 <- unlist(list(tokens_b, sep_token))
    segment_ids2 <- rep(1, length(tokens2))
    tokens <- c(tokens, tokens2)
    segment_ids <- c(segment_ids, segment_ids2)
  }
  input_ids <- convert_tokens_to_ids(tokenizer$vocab, tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask <- rep(1, length(input_ids))

  # Zero-pad up to the sequence length.
  pad_length <- max_seq_length - length(input_ids)
  padding <- rep(0, pad_length)
  input_ids <- c(input_ids, padding)
  input_mask <- c(input_mask, padding)
  segment_ids <- c(segment_ids, padding)

  # Stop now if the lengths aren't right somehow. -JDB
  if (length(input_ids) != max_seq_length |
    length(input_mask) != max_seq_length |
    length(segment_ids) != max_seq_length) {
    stop("input_ids, input_mask, or segment_ids have the wrong length.")
  }

  label_id <- label_map[[example$label]]

  feature <- InputFeatures(
    input_ids = input_ids,
    input_mask = input_mask,
    segment_ids = segment_ids,
    label_id = label_id,
    is_real_example = TRUE
  )
  return(feature)
}



# file_based_convert_examples_to_feature (todo) ---------------------------------

#' Convert a set of \code{InputExample}s to a TFRecord file.
#'
#' description
#'
#' @param examples List of \code{InputExample}s to convert.
#' @param label_list Character (or integer?); possible labels for examples.
#' @param max_seq_length Integer; the maximum number of tokens that will be
#'   considered together.
#' @param tokenizer A tokenizer object to use (e.g. object of class
#'   FullTokenizer).
#' @param output_file Character; path to file to write to.
#'
#' @return return value
#' @export
file_based_convert_examples_to_features <- function(examples,
                                                    label_list,
                                                    max_seq_length,
                                                    tokenizer,
                                                    output_file) {
  stop("file_based_convert_examples_to_features is not yet implemented.")
  # Not sure we need the file_based functions for RBERT MVP. Hold off on
  # implementing for now.
  # tensorflow::tf$python_io$TFRecordWriter()
}


# file_based_input_fn_builder (todo) ---------------------------------------------

#' summary
#'
#' description
#'
#' @param x This parameter will be described when this function is implemented.
#'
#' @return return value
#' @export
file_based_input_fn_builder <- function(x) {
  stop("file_based_input_fn_builder is not yet implemented.")
}

# truncate_seq_pair ------------------------------------------------------

#' Truncate a sequence pair to the maximum length.
#'
#' Truncates a sequence pair to the maximum length.
#' This is a simple heuristic which will always truncate the longer sequence one
#' token at a time (or the first sequence in case of a tie -JDB). This makes
#' more sense than truncating an equal percent of tokens from each, since if one
#' sequence is very short then each token that's truncated likely contains more
#' information than a longer sequence.
#'
#' The python code truncated the sequences in place, using the pass-by-reference
#' functionality of python. In R, we return the truncated sequences in a list.
#'
#' @param tokens_a Character; a vector of tokens in the first input sequence.
#' @param tokens_b Character; a vector of tokens in the second input sequence.
#' @param max_length Integer; the maximum total length of the two sequences.
#'
#' @return A list containing two character vectors: trunc_a and trunc_b.
#' @export
#'
#' @examples
#' \dontrun{
#' tokens_a <- c("a", "b", "c", "d")
#' tokens_b <- c("w", "x", "y", "z")
#' truncate_seq_pair(tokens_a, tokens_b, 5)
#' }
truncate_seq_pair <- function(tokens_a, tokens_b, max_length) {
  trunc_a <- tokens_a
  trunc_b <- tokens_b
  while (TRUE) {
    total_length <- length(trunc_a) + length(trunc_b)
    if (total_length <= max_length) {
      break
    }
    if (length(trunc_a) > length(trunc_b)) {
      trunc_a <- trunc_a[-length(trunc_a)]
    } else {
      trunc_b <- trunc_b[-length(trunc_b)]
    }
  }
  return(list(
    "trunc_a" = trunc_a,
    "trunc_b" = trunc_b
  ))
}


# create_model ------------------------------------------------------------

#' Create a classification model
#'
#' Takes the output layer from a BERT "spine" and appends a classifier layer to
#' it. The output taken from BERT is the pooled first token layers (may want to
#' modify the code to use token-level outputs). The classifier is essentially a
#' single dense layer with softmax.
#'
#' @param bert_config \code{BertConfig} instance.
#' @param is_training Logical; TRUE for training model, FALSE for eval model.
#'   Controls whether dropout will be applied.
#' @param input_ids Integer Tensor of shape \code{[batch_size, seq_length]}.
#' @param input_mask Integer Tensor of shape \code{[batch_size, seq_length]}.
#' @param segment_ids Integer Tensor of shape \code{[batch_size, seq_length]}.
#' @param labels Integer Tensor; represents training example classification
#'   labels. Length = batch size.
#' @param num_labels Integer; number of classification labels.
#' @param use_one_hot_embeddings Logical; whether to use one-hot word embeddings
#'   or tf.embedding_lookup() for the word embeddings.
#'
#' @return A list including the loss (for training) and the model output
#'   (softmax probabilities, log probs).
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'   reuse = tensorflow::tf$AUTO_REUSE
#' ), {
#'   input_ids <- tensorflow::tf$constant(list(
#'     list(31L, 51L, 99L),
#'     list(15L, 5L, 0L)
#'   ))
#'
#'   input_mask <- tensorflow::tf$constant(list(
#'     list(1L, 1L, 1L),
#'     list(1L, 1L, 0L)
#'   ))
#'   token_type_ids <- tensorflow::tf$constant(list(
#'     list(0L, 0L, 1L),
#'     list(0L, 2L, 0L)
#'   ))
#'   config <- BertConfig(
#'     vocab_size = 32000L,
#'     hidden_size = 768L,
#'     num_hidden_layers = 8L,
#'     num_attention_heads = 12L,
#'     intermediate_size = 1024L
#'   )
#'   class_model <- create_model(
#'     bert_config = config,
#'     is_training = TRUE,
#'     input_ids = input_ids,
#'     input_mask = input_mask,
#'     segment_ids = token_type_ids,
#'     labels = c(1L, 2L),
#'     num_labels = 2L,
#'     use_one_hot_embeddings = FALSE
#'   )
#' })
#' }
create_model <- function(bert_config,
                         is_training,
                         input_ids,
                         input_mask,
                         segment_ids,
                         labels,
                         num_labels,
                         use_one_hot_embeddings) {
  model <- BertModel(
    config = bert_config,
    is_training = is_training,
    input_ids = input_ids,
    input_mask = input_mask,
    token_type_ids = segment_ids,
    use_one_hot_embeddings = use_one_hot_embeddings
  )

  # In the demo (BERT colab), we are doing a simple classification task on the
  # entire segment.
  #
  # If you want to use the token-level output, use model$sequence_output
  # instead. (Rather than the pooled_output at the first token. -JDB)
  output_layer <- model$pooled_output

  # Really have to check the indexing every time. -JDB
  # output_layer$shape is a "TensorShape" object that is *zero*-indexed.
  # output_layer$shape$as_list() returns an integer vector, so *one*-indexed.
  # hidden_size <- output_layer$shape[[-1L]]$value # This gives the same as:
  hidden_size <- utils::tail(output_layer$shape$as_list(), -1)

  output_weights <- tensorflow::tf$get_variable(
    name = "output_weights",
    shape = tensorflow::shape(num_labels, hidden_size),
    initializer = tensorflow::tf$truncated_normal_initializer(stddev = 0.02)
  )

  output_bias <- tensorflow::tf$get_variable(
    name = "output_bias",
    shape = tensorflow::shape(num_labels),
    initializer = tensorflow::tf$zeros_initializer()
  )

  with(tensorflow::tf$variable_scope("loss"), {
    if (is_training) {
      # perform 0.1 dropout (keep 90%)
      output_layer <- tensorflow::tf$nn$dropout(output_layer, keep_prob = 0.9)
    }

    logits <- tensorflow::tf$matmul(output_layer,
      output_weights,
      transpose_b = TRUE
    )
    logits <- tensorflow::tf$nn$bias_add(logits, output_bias)
    probabilities <- tensorflow::tf$nn$softmax(logits, axis = -1L)
    log_probs <- tensorflow::tf$nn$log_softmax(logits, axis = -1L)
    one_hot_labels <- tensorflow::tf$one_hot(
      indices = labels,
      depth = num_labels,
      dtype = tensorflow::tf$float32
    )

    # This implies that `labels` has a shape compatible with batch size. Most
    # likely the length of `labels` *is* batch size... Confirm when we get
    # there.
    # I believe that the following is calculating the cross-entropy loss. -JDB
    per_example_loss <- -tensorflow::tf$reduce_sum(one_hot_labels * log_probs,
      axis = -1L
    )
    loss <- tensorflow::tf$reduce_mean(per_example_loss)

    return(list(
      "loss" = loss,
      "per_example_loss" = per_example_loss,
      "logits" = logits,
      "probabilities" = probabilities
    ))
  })
}


# model_fn_builder --------------------------------------------------------

#' Define \code{model_fn} closure for \code{TPUEstimator}
#'
#' Returns \code{model_fn} closure, which is an input to \code{TPUEstimator}.
#'
#' The \code{model_fn} function takes four parameters: \describe{
#' \item{features}{A list (or similar structure) that contains objects such as
#'  \code{input_ids}, \code{input_mask}, \code{segment_ids},  and
#'  \code{label_ids}. These objects will be inputs to the \code{create_model}
#'  function.}
#' \item{labels}{Not used in this function, but presumably we need to
#'  keep this slot here.}
#' \item{mode}{Character; value such as "train", "infer",
#'  or "eval".}
#' \item{params}{Not used in this function, but presumably we need
#'  to keep this slot here.}
#'  }
#'
#' The output of \code{model_fn} is the result of a
#' \code{tf$contrib$tpu$TPUEstimatorSpec} call.
#'
#' This reference may be helpful:
#' \url{https://tensorflow.rstudio.com/tfestimators/articles/creating_estimators.html}
#'
#' @param bert_config \code{BertConfig} instance.
#' @param num_labels Integer; number of classification labels.
#' @param init_checkpoint Character; path to the checkpoint directory, plus
#'   checkpoint name stub (e.g. "bert_model.ckpt"). Path must be absolute and
#'   explicit, starting with "/".
#' @param learning_rate Numeric; the learning rate.
#' @param num_train_steps Integer; number of steps to train for.
#' @param num_warmup_steps Integer; number of steps to use for "warm-up".
#' @param use_tpu Logical; whether to use TPU.
#' @param use_one_hot_embeddings Logical; whether to use one-hot word embeddings
#'   or tf.embedding_lookup() for the word embeddings.
#'
#' @return \code{model_fn} closure for \code{TPUEstimator}.
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'   reuse = tensorflow::tf$AUTO_REUSE
#' ), {
#'   input_ids <- tensorflow::tf$constant(list(
#'     list(31L, 51L, 99L),
#'     list(15L, 5L, 0L)
#'   ))
#'
#'   input_mask <- tensorflow::tf$constant(list(
#'     list(1L, 1L, 1L),
#'     list(1L, 1L, 0L)
#'   ))
#'   token_type_ids <- tensorflow::tf$constant(list(
#'     list(0L, 0L, 1L),
#'     list(0L, 2L, 0L)
#'   ))
#'   config <- BertConfig(
#'     vocab_size = 30522L,
#'     hidden_size = 768L,
#'     num_hidden_layers = 8L,
#'     type_vocab_size = 2L,
#'     num_attention_heads = 12L,
#'     intermediate_size = 3072L
#'   )
#'
#'   temp_dir <- tempdir()
#'   init_checkpoint <- file.path(
#'     temp_dir,
#'     "BERT_checkpoints",
#'     "uncased_L-12_H-768_A-12",
#'     "bert_model.ckpt"
#'   )
#'
#'   example_mod_fn <- model_fn_builder(
#'     bert_config = config,
#'     num_labels = 2L,
#'     init_checkpoint = init_checkpoint,
#'     learning_rate = 0.01,
#'     num_train_steps = 20L,
#'     num_warmup_steps = 10L,
#'     use_tpu = FALSE,
#'     use_one_hot_embeddings = FALSE
#'   )
#' })
#' }
model_fn_builder <- function(bert_config,
                             num_labels,
                             init_checkpoint,
                             learning_rate,
                             num_train_steps,
                             num_warmup_steps,
                             use_tpu,
                             use_one_hot_embeddings) {
  # The `model_fn` for TPUEstimator.
  model_fn <- function(features, labels, mode, params) {
    print("*** Features ***")
    for (name in sort(names(features))) {
      print(paste0(
        "  name = ", name,
        ", shape = ", features[[name]]$shape
      ))
    }

    input_ids <- features$input_ids
    input_mask <- features$input_mask
    segment_ids <- features$segment_ids
    label_ids <- features$label_ids

    # is_real_example = None
    # Come back to confirm this once I know more about what sort of object
    # `features` is. (RBERT issue #25) -JDB
    if ("is_real_example" %in% names(features)) {
      is_real_example <- tensorflow::tf$cast(features$is_real_example,
        dtype = tensorflow::tf$float32
      )
    } else {
      is_real_example <- tensorflow::tf$ones(tensorflow::tf$shape(label_ids),
        dtype = tensorflow::tf$float32
      )
    }
    is_training <- (mode == tensorflow::tf$estimator$ModeKeys$TRAIN)

    created_model <- create_model(
      bert_config = bert_config,
      is_training = is_training,
      input_ids = input_ids,
      input_mask = input_mask,
      segment_ids = segment_ids,
      labels = label_ids,
      num_labels = num_labels,
      use_one_hot_embeddings = use_one_hot_embeddings
    )
    total_loss <- created_model$loss
    per_example_loss <- created_model$per_example_loss
    logits <- created_model$logits
    probabilities <- created_model$probabilities

    tvars <- tensorflow::tf$trainable_variables()
    initialized_variable_names <- list()
    scaffold_fn <- NULL
    if (!is.null(init_checkpoint)) {
      gamap <- get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      assignment_map <- gamap$assignment_map
      initialized_variable_names <- gamap$initialized_variable_names
      if (use_tpu) {
        tpu_scaffold <- function() {
          tensorflow::tf$train$init_from_checkpoint(
            init_checkpoint,
            assignment_map
          )
          return(tensorflow::tf$train$Scaffold())
        }
        scaffold_fn <- tpu_scaffold
      } else {
        tensorflow::tf$train$init_from_checkpoint(
          init_checkpoint,
          assignment_map
        )
      }
    }
    print("*** Trainable Variables ***")
    for (var in tvars) {
      init_string <- ""
      if (var$name %in% initialized_variable_names) {
        init_string <- ", *INIT_FROM_CKPT*"
      }
      print(paste0(
        "  name = ", var$name,
        ", shape = ", var$shape,
        init_string
      ))
    }

    output_spec <- NULL
    if (is_training) {
      train_op <- create_optimizer(
        total_loss,
        learning_rate,
        num_train_steps,
        num_warmup_steps,
        use_tpu
      )
      output_spec <- tensorflow::tf$contrib$tpu$TPUEstimatorSpec(
        mode = mode,
        loss = total_loss,
        train_op = train_op,
        scaffold_fn = scaffold_fn
      )
    } else if (mode == tensorflow::tf$estimator$ModeKeys$EVAL) {
      metric_fn <- function(per_example_loss,
                            label_ids,
                            logits,
                            is_real_example) {
        predictions <- tensorflow::tf$argmax(logits,
          axis = -1L,
          output_type = tensorflow::tf$int32
        )
        accuracy <- tensorflow::tf$metrics$accuracy(
          labels = label_ids,
          predictions = predictions,
          weights = is_real_example
        )
        loss <- tensorflow::tf$metrics$mean(
          values = per_example_loss,
          weights = is_real_example
        )
        return(list(
          "eval_accuracy" = accuracy,
          "eval_loss" = loss
        ))
      }
      # "`eval_metrics` is a tuple of `metric_fn` and `tensors`..."
      # See link in comments below. -JDB
      eval_metrics <- list(
        metric_fn,
        list(
          per_example_loss,
          label_ids,
          logits,
          is_real_example
        )
      )
      output_spec <- tensorflow::tf$contrib$tpu$TPUEstimatorSpec(
        mode = mode,
        loss = total_loss,
        eval_metrics = eval_metrics,
        scaffold_fn = scaffold_fn
      )
    } else {
      # `predictions`: Predictions `Tensor` or dict of `Tensor`.
      # It appears that using a list, as I have below, works; see:
      # https://tensorflow.rstudio.com/tfestimators/articles/creating_estimators.html
      # -JDB
      output_spec <- tensorflow::tf$contrib$tpu$TPUEstimatorSpec(
        mode = mode,
        predictions = list("probabilities" = probabilities),
        scaffold_fn = scaffold_fn
      )
    }
    return(output_spec)
  } # end of `model_fn` definition

  return(model_fn)
}


# input_fn_builder --------------------------------------------------------

#' Create an \code{input_fn} closure to be passed to TPUEstimator
#'
#' Creates an \code{input_fn} closure to be passed to TPUEstimator. The output
#' of this closure is the (modified) output of
#' \code{tensorflow::tf$data$Dataset$from_tensor_slices} (an object of class
#' "tensorflow.python.data.ops.dataset_ops.BatchDataset").
#'
#' @param features A list of features (objects of class \code{InputFeatures}).
#' @param seq_length Integer; the maximum length (number of tokens) of each
#'   example. (Examples should already be padded to this length by this point.)
#' @param is_training Logical; whether these are training examples.
#' @param drop_remainder Logical; whether to drop the extra if the number of
#'   elements in the dataset is not an exact multiple of the batch size,
#'
#' @return An \code{input_fn} closure to be passed to TPUEstimator.
#' @export
#'
#' @examples
#' \dontrun{
#' tokenizer <- FullTokenizer("vocab.txt")
#' seq_len <- 15L
#' input_ex1 <- InputExample(
#'   guid = 1L,
#'   text_a = "Some text to classify.",
#'   text_b = "More wordy words.",
#'   label = "good"
#' )
#' input_ex2 <- InputExample(
#'   guid = 2L,
#'   text_a = "This is another example.",
#'   text_b = "So many words.",
#'   label = "bad"
#' )
#' feat <- convert_examples_to_features(
#'   examples = list(input_ex1, input_ex2),
#'   label_list = c("good", "bad"),
#'   max_seq_length = seq_len,
#'   tokenizer = tokenizer
#' )
#' input_fn <- input_fn_builder(
#'   features = feat,
#'   seq_length = seq_len,
#'   is_training = TRUE,
#'   drop_remainder = FALSE
#' )
#' }
input_fn_builder <- function(features,
                             seq_length,
                             is_training,
                             drop_remainder) {
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
  all_segment_ids <- purrr::map(
    features,
    function(f) {
      as.integer(f$segment_ids)
    }
  )
  all_label_ids <- purrr::map(
    features,
    function(f) {
      as.integer(f$label_id)
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
        "segment_ids" = tensorflow::tf$constant(
          all_segment_ids,
          shape = tensorflow::shape(num_examples, seq_length),
          dtype = tensorflow::tf$int32
        ),
        "label_ids" = tensorflow::tf$constant(
          all_label_ids,
          shape = tensorflow::shape(num_examples), # check
          dtype = tensorflow::tf$int32
        )
      )
    )

    if (is_training) {
      # "The default behavior (if `count` is `None` or `-1`) is for the dataset
      # be repeated indefinitely." ('indefinitely' sounds like a lot. -JDB)

      # Note the back ticks required to avoid colliding with the 'repeat'
      # control word in R. -JDB
      d <- d$`repeat`()
      d <- d$shuffle(buffer_size = 100L)
    }
    # debugging... -JDB
    if (is.null(batch_size)) {
      print("null batch size; defaulting to 32")
      batch_size <- 32L
    }
    d <- d$batch(
      batch_size = batch_size,
      drop_remainder = drop_remainder
    )
    return(d) # return from `input_fn`
  }
  return(input_fn)
}


# convert_examples_to_features --------------------------------------------

#' Convert \code{InputExample}s to \code{InputFeatures}
#'
#' Converts a set of \code{InputExample}s to a list of \code{InputFeatures}.
#'
#' @param examples List of \code{InputExample}s to convert.
#' @param label_list Character (or integer?); possible labels for examples.
#' @param max_seq_length Integer; the maximum number of tokens that will be
#'   considered together.
#' @param tokenizer A tokenizer object to use (e.g. object of class
#'   FullTokenizer).
#'
#' @return A list of \code{InputFeatures}.
#' @export
#'
#' @examples
#' \dontrun{
#' tokenizer <- FullTokenizer("vocab.txt")
#' input_ex1 <- InputExample(
#'   guid = 1L,
#'   text_a = "Some text to classify.",
#'   text_b = "More wordy words.",
#'   label = "good"
#' )
#' input_ex2 <- InputExample(
#'   guid = 2L,
#'   text_a = "This is another example.",
#'   text_b = "So many words.",
#'   label = "bad"
#' )
#' feat <- convert_examples_to_features(
#'   examples = list(input_ex1, input_ex2),
#'   label_list = c("good", "bad"),
#'   max_seq_length = 15L,
#'   tokenizer = tokenizer
#' )
#' }
convert_examples_to_features <- function(examples,
                                         label_list,
                                         max_seq_length,
                                         tokenizer) {
  example_indices <- seq_along(examples)
  num_examples <- length(examples)
  features <- purrr::map2(
    example_indices,
    examples,
    function(ex_index, example,
             label_list, max_seq_length, tokenizer) {
      convert_single_example(
        ex_index = ex_index,
        example = example,
        label_list = label_list,
        max_seq_length = max_seq_length,
        tokenizer = tokenizer
      )
    },
    label_list, max_seq_length, tokenizer
  )
  return(features)
}
