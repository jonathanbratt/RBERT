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

# find checkpoint files ---------------------------------------------------


#' Find Checkpoint Files
#'
#' Given the path to a checkpoint directory, return the paths to certain files
#' in that directory.
#'
#' @param ckpt_dir Character; the path to the checkpoint directory. If this
#'   argument is NULL, the associated functions also return NULL.
#' @name find_files
NULL

#' @describeIn find_files Find the vocabulary file ('vocab.txt').
#' @export
find_vocab <- function(ckpt_dir) {
  # If this gets called for a NULL ckpt_dir, return NULL.
  if (is.null(ckpt_dir)) {
    return(NULL)
  }

  vocab_file <- file.path(ckpt_dir, "vocab.txt")
  if (file.exists(vocab_file)) {
    return(vocab_file)
  } else {
    stop("No file named 'vocab.txt' found in ", ckpt_dir) # nocov
  }
}


#' @describeIn find_files Find the config file ('bert_config.json').
#' @export
find_config <- function(ckpt_dir) {
  # If this gets called for a NULL ckpt_dir, return NULL.
  if (is.null(ckpt_dir)) {
    return(NULL)
  }

  config_file <- file.path(ckpt_dir, "bert_config.json")
  if (file.exists(config_file)) {
    return(config_file)
  } else {
    stop("No file named 'bert_config.json' found in ", ckpt_dir) # nocov
  }
}

#' @describeIn find_files Find the checkpoint file stub (files begin with
#'   'bert_model.ckpt').
#' @export
find_ckpt <- function(ckpt_dir) {
  # If this gets called for a NULL ckpt_dir, return NULL.
  if (is.null(ckpt_dir)) {
    return(NULL)
  }

  # The path we want to return here isn't an actual file, but a name stub for
  # files with suffixes '.index', '.meta', etc.
  ckpt_filestub <- file.path(ckpt_dir, "bert_model.ckpt")
  ckpt_file1 <- file.path(ckpt_dir, "bert_model.ckpt.index")
  # Used to check for .meta file, but that's not in every checkpoint.
  if (file.exists(ckpt_file1)) {
    return(ckpt_filestub)
  } else {
    stop("Checkpoint file(s) missing from ", ckpt_dir) # nocov
  }
}

#' Find Paths to Checkpoint Files
#'
#' In some functions, the user can specify a model, a ckpt_dir, and/or specific
#' paths to checkpoint files. This function sorts all of that out.
#'
#'  This used to inheritParams from extract_features. FIX!
#' @return A list with components vocab_file, bert_config_file, and
#'   init_checkpoint.
#' @keywords internal
.infer_model_paths <- function(model = c(
  "bert_tiny_uncased",
  "bert_mini_uncased",
  "bert_small_uncased",
  "bert_medium_uncased",
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
init_checkpoint = find_ckpt(ckpt_dir)) {
  # Deal with the fact that model will never be *missing* when this function is
  # called, but we don't want the calling functions to have to deal with parsing
  # the argument.
  if (length(model) > 1) {
    model <- NULL
  } else {
    model <- match.arg(model)
  }

  # If any of the necessary files aren't specified, try to find them. This would
  # most likely only happen if they specified one file but not all (and left
  # ckpt_dir as NULL), but run this to be sure.
  vocab_file <- vocab_file %||% find_vocab(ckpt_dir)
  bert_config_file <- bert_config_file %||% find_config(ckpt_dir)
  init_checkpoint <- init_checkpoint %||% find_ckpt(ckpt_dir)

  # At this point either we have the paths, or we need to infer from the model.
  if ((is.null(vocab_file) |
       is.null(bert_config_file) |
       is.null(init_checkpoint))) {
    if (is.null(model)) {
      stop(
        "You must specify a model, a ckpt_dir, or the locations of ",
        "vocab_file, bert_config_file, and init_checkpoint."
      )
    } else {
      dir <- .choose_BERT_dir(NULL)
      ckpt_dir <- .get_model_subdir(model, dir)
      .maybe_download_checkpoint(
        model = model,
        dir = dir,
        ckpt_dir = ckpt_dir
      )

      # If we made it here, they have the model, so set the file locations.
      vocab_file <- find_vocab(ckpt_dir)
      bert_config_file <- find_config(ckpt_dir)
      init_checkpoint <- find_ckpt(ckpt_dir)
    }
  }
  return(
    list(
      vocab_file = vocab_file,
      bert_config_file = bert_config_file,
      init_checkpoint = init_checkpoint
    )
  )
}

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
.transpose_for_scores <- function(input_tensor,
                                  batch_size,
                                  num_attention_heads,
                                  seq_length,
                                  width) {
  # NB: the element ordering convention used by TF is different from the
  # convention used by, say, as.array in R.
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


