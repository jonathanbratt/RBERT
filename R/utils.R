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
  ckpt_file2 <- file.path(ckpt_dir, "bert_model.ckpt.meta")
  if (file.exists(ckpt_file1) & file.exists(ckpt_file2)) {
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
#' @inheritParams extract_features
#' @return A list with components vocab_file, bert_config_file, and
#'   init_checkpoint.
#' @keywords internal
.infer_model_paths <- function(model = c(
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
