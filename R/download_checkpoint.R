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
#'
#' @param model Character vector. Which model checkpoint to download.
#' @param dir Character vector. Destination directory for checkpoint. Leave
#'   `NULL` to allow `RBERT` to automatically choose a directory. The path is
#'   determined from the `dir` parameter if supplied, followed by the
#'   `RBERT.dir` option (set using \link{set_BERT_dir}), followed by an "RBERT"
#'   folder in the user cache directory (determined using
#'   \code{\link[rappdirs]{user_cache_dir()}}). If you provide a `dir`, the
#'   `RBERT.dir` option will be updated to that location.
#' @param url Character vector. An optional url from which to download a
#'   checkpoint. Overrides \code{model} parameter if not NULL.
#' @param force Logical. Download even if the checkpoint already exists in the
#'   specified directory? Default \code{FALSE}.
#' @param keep_zip Logical. Keep the zip file? Leave as \code{FALSE} to save
#'   space.
#'
#' @return If successful, returns the path to the downloaded checkpoint.
#' @export
#'
#' @examples
#' \dontrun{
#'  download_BERT_checkpoint("bert_base_uncased")
#'  download_BERT_checkpoint("bert_large_uncased")
#'  temp_dir <- tempdir()
#'  download_BERT_checkpoint("bert_base_uncased", dir = temp_dir)
#' }
download_BERT_checkpoint <- function(model = c("bert_base_uncased",
                                               "bert_base_cased",
                                               "bert_large_uncased",
                                               "bert_large_cased",
                                               "bert_large_uncased_wwm",
                                               "bert_large_cased_wwm",
                                               "bert_base_multilingual_cased",
                                               "bert_base_chinese"),
                                     dir = NULL,
                                     url = NULL,
                                     force = FALSE,
                                     keep_zip = FALSE) {
  if (is.null(url)) {
    model <- match.arg(model)
    url <- .get_model_url(model)
  }

  dir <- .choose_BERT_dir(dir)

  # Use the same location for the dir for the rest of this session. This
  # function also attempts to create the directory if it does not exist.
  set_BERT_dir(dir)

  # This may have to get more complicated as we add new checkpoints.
  checkpoint_subdir <- normalizePath(
    file.path(dir, stringr::str_extract(url, "[^/]+(?=\\.zip)")),
    mustWork = FALSE
  )
  checkpoint_zip_path <- paste0(checkpoint_subdir, ".zip")

  has_checkpoint <- .has_checkpoint(model = model, dir = dir, url = url)

  if (
    force ||
    (keep_zip && !file.exists(checkpoint_zip_path)) ||
    !has_checkpoint
  ) {
    .download_BERT_checkpoint(url, checkpoint_zip_path)
  }

  if (force || !has_checkpoint) {
    .process_BERT_checkpoint(dir, checkpoint_zip_path)
  }

  if (!keep_zip && file.exists(checkpoint_zip_path)) {
    file.remove(checkpoint_zip_path)
  }

  # The normalizePath shouldn't be necessary here, but I was getting
  # inconsistent returns on Windows. I suspect it's because the return is
  # slightly different when the path exists.
  return(normalizePath(checkpoint_subdir))
}

#' Choose a directory for BERT checkpoints
#'
#' If `dir` is not NULL, this function simply returns `dir`. Otherwise it checks
#' the `RBERT.dir` param, and then uses \code{\link[rappdirs]{user_cache_dir}}
#' to choose a directory if necessary.
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return A character vector indicating a directory in which BERT checkpoints
#'   are stored.
#' @keywords internal
.choose_BERT_dir <- function(dir) {
  return(
    dir %||%
      getOption("RBERT.dir") %||%
      rappdirs::user_cache_dir(appname = "RBERT")
  )
}

#' Check whether the user already has a checkpoint
#'
#' Check the specified dir (or the default dir if none is specified) for a given
#' model or url.
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return A logical indicating whether the user already has that checkpoint in
#'   that location.
#' @keywords internal
.has_checkpoint <- function(model = c("bert_base_uncased",
                                      "bert_base_cased",
                                      "bert_large_uncased",
                                      "bert_large_cased",
                                      "bert_large_uncased_wwm",
                                      "bert_large_cased_wwm",
                                      "bert_base_multilingual_cased",
                                      "bert_base_chinese"),
                            dir = NULL,
                            url = NULL) {
  dir <- .choose_BERT_dir(dir)
  if (is.null(url)) {
    model <- match.arg(model)
    url <- .get_model_url(model)
  }
  checkpoint_subdir <- normalizePath(
    file.path(dir, stringr::str_extract(url, "[^/]+(?=\\.zip)")),
    mustWork = FALSE
  )

  filenames <- list.files(checkpoint_subdir)

  return(
    !any(
      !any(grepl("bert_config.json", filenames), na.rm = TRUE),
      !any(grepl("vocab.txt", filenames), na.rm = TRUE),
      !any(grepl("bert_model.ckpt", filenames), na.rm = TRUE)
    )
  )
}

#' Download a checkpoint zip file
#'
#' @inheritParams download_BERT_checkpoint
#' @param checkpoint_zip_path The path to which the checkpoint zip should be
#'   downloaded.
#'
#' @return `TRUE` invisibly.
#' @keywords internal
.download_BERT_checkpoint <- function(url, checkpoint_zip_path) {
  status <- utils::download.file(
    url = url,
    destfile = checkpoint_zip_path,
    method = "libcurl"
  )
  if (status != 0) {
    stop("Checkpoint download failed.")  # nocovr
  }
  invisible(TRUE)
}

#' Unzip and check a BERT checkpoint zip
#'
#' @inheritParams download_BERT_checkpoint
#' @inheritParams .download_BERT_checkpoint
#'
#' @return `TRUE` invisibly.
#' @keywords internal
.process_BERT_checkpoint <- function(dir, checkpoint_zip_path, keep_zip) {
  # We're only here if the files don't exist or we're supposed to overwrite, so
  # always overwrite.
  filenames <- utils::unzip(
    zipfile = checkpoint_zip_path,
    exdir = dir,
    overwrite = TRUE
  )

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

  invisible(TRUE)
}



# .get_model_url ------------------------------------------------

#' Get url of a BERT checkpoint
#'
#' Returns the url of the specified BERT checkpoint from the Google Research
#' collection.
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return The url to the specified Google Research BERT model.
#' @keywords internal
.get_model_url <- function(model = c("bert_base_uncased",
                                    "bert_base_cased",
                                    "bert_large_uncased",
                                    "bert_large_cased",
                                    "bert_large_uncased_wwm",
                                    "bert_large_cased_wwm",
                                    "bert_base_multilingual_cased",
                                    "bert_base_chinese")) {
  root_url <- "https://storage.googleapis.com/bert_models/"
  checkpoint_zips <- c(
    "bert_base_uncased" = "2018_10_18/uncased_L-12_H-768_A-12.zip",
    "bert_base_cased" = "2018_10_18/cased_L-12_H-768_A-12.zip",
    "bert_large_uncased" = "2018_10_18/uncased_L-24_H-1024_A-16.zip",
    "bert_large_cased" = "2018_10_18/cased_L-24_H-1024_A-16.zip",
    "bert_large_uncased_wwm" = "2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
    "bert_large_cased_wwm" = "2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
    "bert_base_multilingual_cased" =
      "2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    "bert_base_chinese" = "2018_11_03/chinese_L-12_H-768_A-12.zip"
  )

  checkpoint_url_map <- paste0(
    root_url,
    checkpoint_zips
  )
  names(checkpoint_url_map) <- names(checkpoint_zips)

  model <- match.arg(model)
  return(checkpoint_url_map[[model]])
}


# set_BERT_dir ------------------------------------------------------------

#' Set the directory for BERT checkpoints
#'
#' Set a given `dir` as the default BERT checkpoint directory for this session,
#' and create it if it does not exist.
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return A list with the previous value of `BERT.dir` (invisibly).
#' @export
#'
#' @examples
#' \dontrun{
#'   set_BERT_dir("fake_dir")
#' }
set_BERT_dir <- function(dir) {
  if (!file.exists(dir)) {
    dir.create(dir) # nocov
  }
  options(BERT.dir = dir)
}


#' Default value for NULL
#'
#' Copied from `rlang` to avoid importing that package.
#'
#' @param x, y If x is NULL, will return y; otherwise returns x.
#'
#' @return x or y.
#' @keywords internal
`%||%` <- function (x, y) {
  if (is.null(x))
    y
  else x
}
