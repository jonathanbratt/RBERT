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

# download_BERT_checkpoint ------------------------------------------------

#' Download a BERT checkpoint
#'
#' Downloads the specified BERT checkpoint from the Google Research collection.
#' Will create a "BERT_checkpoints" directory in the specified location (user's
#' home directory, by default) if not already present.
#'
#' @param model Character vector; which model checkpoint to download.
#' @param destination Character vector; destination directory for checkpoint.
#' @param url Character vector; a url from which to download a checkpoint.
#'   Overrides \code{model} parameter if not NULL.
#' @param overwrite Logical; parameter passed along to \code{utils::unzip}.
#'
#' @return If successful, returns the path to the downloaded checkpoint.
#' @export
#'
#' @examples
#' \dontrun{
#'  download_BERT_checkpoint("bert_base_uncased", destination = "/shared/")
#'  download_BERT_checkpoint("bert_large_uncased", destination = "/shared/")
#' }
download_BERT_checkpoint <- function(model = c("bert_base_uncased",
                                               "bert_base_cased",
                                               "bert_large_uncased",
                                               "bert_large_cased",
                                               "bert_large_uncased_wwm",
                                               "bert_large_cased_wwm",
                                               "bert_base_multilingual_cased",
                                               "bert_base_chinese"),
                                     destination = "~",
                                     url = NULL,
                                     overwrite = TRUE) {
  if (is.null(url)) {
    model <- match.arg(model)
    url <- get_model_url(model)
  }

  checkpoint_dir <- file.path(destination, "BERT_checkpoints")
  if (!file.exists(checkpoint_dir)) {
    dir.create(checkpoint_dir)
  }
  checkpoint_zip <- tempfile(pattern = "bert_",
                             fileext = ".zip")

  status <- utils::download.file(url = url,
                                 destfile = checkpoint_zip,
                                 method = "libcurl")
  if (status != 0) {
    stop("Checkpoint download failed.")  # nocovr
  }

  filenames <- utils::unzip(zipfile = checkpoint_zip,
                            exdir = checkpoint_dir,
                            overwrite = overwrite)

  # Quick check to see if expected files found.
  if (!any(grepl("bert_config.json", filenames))) {
    warning("No bert_config file found.")  # nocovr
  }
  if (!any(grepl("vocab.txt", filenames))) {
    warning("No vocabulary file found.")  # nocovr
  }
  if (!any(grepl("bert_model.ckpt", filenames))) {
    warning("No checkpoint file found.")  # nocovr
  }
  file.remove(checkpoint_zip)

  # Return the path to the checkpoint directory.
  return(
    normalizePath(
      dirname(filenames[[1]])
    )
  )

}

# get_model_url ------------------------------------------------

#' Get url of a BERT checkpoint
#'
#' Returns the url of the specified BERT checkpoint from the Google Research
#' collection.
#'
#' @param model Character vector; which model checkpoint to retrieve.
#'
#' @return The url to the specified Google Research BERT model.
#' @keywords internal
get_model_url <- function(model = c("bert_base_uncased",
                                    "bert_base_cased",
                                    "bert_large_uncased",
                                    "bert_large_cased",
                                    "bert_large_uncased_wwm",
                                    "bert_large_cased_wwm",
                                    "bert_base_multilingual_cased",
                                    "bert_base_chinese")) {
  checkpoint_url_map <- c(
    "bert_base_uncased" = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "bert_base_cased" = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
    "bert_large_uncased" = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
    "bert_large_cased" = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip",
    "bert_large_uncased_wwm" = "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
    "bert_large_cased_wwm" = "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
    "bert_base_multilingual_cased" = "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    "bert_base_chinese" = "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"
  )

  model <- match.arg(model)
  return(checkpoint_url_map[[model]])
}

