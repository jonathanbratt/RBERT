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
#' Downloads the specified BERT checkpoint from the Google Research collection
#' or other repositories.
#'
#' @section Checkpoints: `download_BERT_checkpoint` knows about several
#'   pre-trained BERT checkpoints. You can specify these checkpoints using the
#'   `model` parameter. Alternatively, you can supply a direct `url` to any BERT
#'   tensorflow checkpoint.
#'
#'   \tabular{rccccl}{ model \tab layers \tab hidden \tab heads \tab parameters
#'   \tab special\cr bert_base_* \tab 12 \tab 768 \tab 12 \tab 110M\cr
#'   bert_large_* \tab 24 \tab 1024 \tab 16 \tab 340M\cr bert_large_*_wwm \tab
#'   24 \tab 1024 \tab 16 \tab 340M \tab whole word masking\cr
#'   bert_base_multilingual_cased \tab 12 \tab 768 \tab 12 \tab 110M \tab 104
#'   languages\cr bert_base_chinese \tab 12 \tab 768 \tab 12 \tab 110M \tab
#'   Chinese Simplified and Traditional\cr scibert_scivocab_* \tab 12 \tab 768
#'   \tab 12 \tab 110M \tab Trained using the full text of 1.14M scientific
#'   papers (18\% computer science, 82\% biomedical), with a science-specific
#'   vocabulary.\cr scibert_basevocab_uncased \tab 12 \tab 768 \tab 12 \tab 110M
#'   \tab As scibert_scivocab_*, but using the original BERT vocabulary. }
#'
#' @param model Character vector. Which model checkpoint to download.
#' @param dir Character vector. Destination directory for checkpoints. Leave
#'   `NULL` to allow `RBERT` to automatically choose a directory. The path is
#'   determined from the `dir` parameter if supplied, followed by the
#'   `RBERT.dir` option (set using \link{set_BERT_dir}), followed by an "RBERT"
#'   folder in the user cache directory (determined using
#'   \code{\link[rappdirs]{user_cache_dir}}). If you provide a `dir`, the
#'   `RBERT.dir` option will be updated to that location. Note that the
#'   checkpoint will create a subdirectory inside this `dir`.
#' @param url Character vector. An optional url from which to download a
#'   checkpoint. Overrides \code{model} parameter if not NULL.
#' @param force Logical. Download even if the checkpoint already exists in the
#'   specified directory? Default \code{FALSE}.
#' @param keep_archive Logical. Keep the zip (or other archive) file? Leave as
#'   \code{FALSE} to save space.
#' @param archive_type How is the checkpoint archived? We currently support
#'   "zip" and "tar-gzip". Leave NULL to infer from the `url`.
#'
#' @return If successful, returns the path to the downloaded checkpoint.
#' @export
#'
#' @source \url{https://github.com/google-research/bert}
#'   \url{https://github.com/allenai/scibert}
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
                                               "bert_base_chinese",
                                               "scibert_scivocab_uncased",
                                               "scibert_scivocab_cased",
                                               "scibert_basevocab_uncased",
                                               "scibert_basevocab_cased"),
                                     dir = NULL,
                                     url = NULL,
                                     force = FALSE,
                                     keep_archive = FALSE,
                                     archive_type = NULL) {
  dir <- .choose_BERT_dir(dir)
  # Use the same location for the dir for the rest of this session. This
  # function also attempts to create the directory if it does not exist.
  set_BERT_dir(dir)

  if (is.null(url)) {
    model <- match.arg(model)
    url <- .get_model_url(model)
    archive_type <- .get_model_archive_type(model)
    checkpoint_subdir <- .get_model_subdir(model, dir)
    checkpoint_archive_path <- .get_model_archive_path(model, dir, archive_type)
  } else {
    model <- NULL
    archive_type <- archive_type %||% .infer_archive_type(url)
    checkpoint_subdir <- .infer_checkpoint_subdir(url, dir)
    checkpoint_archive_path <- .infer_checkpoint_archive_path(url, dir)
  }

  has_checkpoint <- .has_checkpoint(
    model = model,
    dir = dir,
    checkpoint_subdir = checkpoint_subdir
  )

  if (
    force ||
    (keep_archive && !file.exists(checkpoint_archive_path)) ||
    !has_checkpoint
  ) {
    .download_BERT_checkpoint(url, checkpoint_archive_path)
  }

  if (force || !has_checkpoint) {
    .process_BERT_checkpoint(
      dir,
      checkpoint_archive_path,
      checkpoint_subdir,
      archive_type
    )
  }

  if (!keep_archive && file.exists(checkpoint_archive_path)) {
    file.remove(checkpoint_archive_path)
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
      getOption("BERT.dir") %||%
      rappdirs::user_cache_dir(appname = "RBERT")
  )
}

#' Check whether the user already has a checkpoint
#'
#' Check the specified dir (or the default dir if none is specified) for a given
#' model or url.
#'
#' @inheritParams download_BERT_checkpoint
#' @param checkpoint_subdir The path to the subdir where this checkpoint should
#'   be saved. If model is given, checkpoint_subdir is inferred.
#'
#' @return A logical indicating whether the user already has that checkpoint in
#'   that location.
#' @keywords internal
.has_checkpoint <- function(model = NULL,
                            dir = NULL,
                            checkpoint_subdir = NULL) {
  dir <- .choose_BERT_dir(dir)
  if (is.null(checkpoint_subdir)) {
    checkpoint_subdir <- .get_model_subdir(model, dir)
  }
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
.process_BERT_checkpoint <- function(dir,
                                     checkpoint_archive_path,
                                     checkpoint_subdir,
                                     archive_type) {
  # We're only here if the files don't exist or we're supposed to overwrite, so
  # always overwrite.
  switch(
    archive_type,
    "zip" = utils::unzip(
      zipfile = checkpoint_archive_path,
      exdir = checkpoint_subdir,
      overwrite = TRUE
    ),
    "tar-gzip" = {
      con <- gzfile(checkpoint_archive_path, open = "rb")
      utils::untar(
        con,
        exdir = checkpoint_subdir
      )
      close(con)
    }
  )

  # We write into the subdir, but *usually* it'll make a folder inside of that
  # dir. Move everything up to be inside checkpoint_subdir.
  extra_dirs <- list.dirs(
    checkpoint_subdir, full.names = TRUE, recursive = FALSE
  )
  if (length(extra_dirs) > 0) {
    for(dir_name in extra_dirs) {
      cp_files <- list.files(
        dir_name,
        recursive = TRUE
      )
      file.rename(
        file.path(dir_name, cp_files),
        file.path(checkpoint_subdir, cp_files)
      )
      unlink(dir_name, recursive = TRUE)
    }
  }

  filenames <- list.files(checkpoint_subdir)

  # Quick check to see if expected files found.
  if (!("bert_config.json" %in% filenames)) {
    warning("No bert_config file found.")  # nocovr
  }
  if (!("vocab.txt" %in% filenames)) {
    warning("No vocabulary file found.")  # nocovr
  }
  if (!any(grepl("bert_model.ckpt", filenames))) {
    warning("No checkpoint file found.")  # nocovr
  }

  invisible(TRUE)
}



# .get_model_* ------------------------------------------------

#' Get url of a BERT checkpoint
#'
#' Returns the url of the specified BERT checkpoint from the Google Research
#' collection or other repository.
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return The url to the specified BERT model.
#' @keywords internal
.get_model_url <- function(model) {
  return(
    checkpoint_url_map[checkpoint_url_map$model == model,][["url"]]
  )
}

#' Get archive type of a BERT checkpoint
#'
#' Returns the archive type ("zip" or "tar-gzip") of the specified BERT
#' checkpoint from the Google Research collection or other repository.
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return The archive type to the specified BERT model.
#' @keywords internal
.get_model_archive_type <- function(model) {
  return(
    checkpoint_url_map[checkpoint_url_map$model == model,][["archive_type"]]
  )
}

#' Locate a subdir for a BERT checkpoint
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return The path to the sub-directory where the checkpoint should be saved.
#' @keywords internal
.get_model_subdir <- function(model, dir) {
  return(
    normalizePath(
      file.path(dir, model),
      mustWork = FALSE
    )
  )
}

#' Locate an archive file for a BERT checkpoint
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return The path to the archive file where the raw checkpoint should be
#'   saved.
#' @keywords internal
.get_model_archive_path <- function(model, dir, archive_type) {
  archive_ending <- c(
    "zip" = ".zip",
    "tar-gzip" = ".tar.gz"
  )[[archive_type]]
  return(
    normalizePath(
      file.path(dir, paste0(model, archive_ending)),
      mustWork = FALSE
    )
  )
}


# .infer_archive_* --------------------------------------------------------

#' Infer the archive type for a BERT checkpoint
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return A character vector, currently either "zip" or "tar-gzip".
#' @keywords internal
.infer_archive_type <- function(url) {
  if (stringr::str_detect(url, "\\.tar\\.gz$")) {
    return("tar-gzip")
  } else if (stringr::str_detect(url, "\\.zip$")) {
    return("zip")
  } else { # nocov start
    stop(
      "Unknown archive type. Please supply an explicit archive_type."
    )
  } # nocov end
}

#' Infer the subdir for a BERT checkpoint
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return A character vector file path, reflecting the "name" part of a
#'   checkpoint `url`, placed within `dir`.
#' @keywords internal
.infer_checkpoint_subdir <- function(url, dir) {
  return(
    normalizePath(
      file.path(
        dir,
        stringr::str_replace_all(
          basename(url),
          c(
            "\\.tar\\.gz$" = "",
            "\\.zip$" = ""
          )
        )
      ),
      mustWork = FALSE
    )
  )
}

#' Infer the path to the archive for a BERT checkpoint
#'
#' @inheritParams download_BERT_checkpoint
#'
#' @return A character vector file path, pointing to where the raw checkpoint
#'   archive should be saved.
#' @keywords internal
.infer_checkpoint_archive_path <- function(url, dir) {
  return(
    normalizePath(
      file.path(
        dir,
        basename(url)
      ),
      mustWork = FALSE
    )
  )
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
  dir <- normalizePath(dir)
  options(BERT.dir = dir)
}

# Copied from `rlang` to avoid importing that package. Roxygen doesn't like it
# and I'm not sure how to fix that, so install I'm not documenting.
`%||%` <- function (x, y) {
  if (is.null(x))
    y
  else x
}
