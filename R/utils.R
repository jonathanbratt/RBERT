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
#' @param cp_dir Character; the path to the checkpoint directory.
#' @name find_files
#' @keywords internal
NULL

#' @describeIn find_files Find the vocabulary file ('vocab.txt').
#' @keywords internal
.find_vocab <- function(cp_dir) {
  vocab_file <- file.path(cp_dir, 'vocab.txt')
  if (file.exists(vocab_file)) {
    return(vocab_file)
  } else {
    stop("No file named 'vocab.txt' found in ", cp_dir) # nocov
  }
}


#' @describeIn find_files Find the config file ('bert_config.json').
#' @keywords internal
.find_config <- function(cp_dir) {
  config_file <- file.path(cp_dir, 'bert_config.json')
  if (file.exists(config_file)) {
    return(config_file)
  } else {
    stop("No file named 'bert_config.json' found in ", cp_dir) # nocov
  }
}

#' @describeIn find_files Find the checkpoint file stub (files begin with
#'   'bert_model.ckpt').
#' @keywords internal
.find_ckpt <- function(cp_dir) {
  # The path we want to return here isn't an actual file, but a name stub for
  # files with suffixes '.index', '.meta', etc.
  ckpt_filestub <- file.path(cp_dir, 'bert_model.ckpt')
  ckpt_file1 <- file.path(cp_dir, 'bert_model.ckpt.index')
  ckpt_file2 <- file.path(cp_dir, 'bert_model.ckpt.meta')
  if (file.exists(ckpt_file1) & file.exists(ckpt_file1)) {
    return(ckpt_filestub)
  } else {
    stop("Checkpoint file(s) missing from ", cp_dir) # nocov
  }
}
