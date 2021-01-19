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



# tokenize_input ----------------------------------------------------

#' Tokenize Text for Input to BERT
#'
#' Given a list of text sequences and a wordpiece vocabulary, tokenize the input
#' sequences into a form suitable for use with RBERT. If the input is a flat
#' list or character vector, the sequences will be single-segment. If the input
#' contains length-2 sublists or vectors, those examples will be two-segment
#' sequences, e.g. for doing sentence-pair classification.
#'
#' The token type ids (0 for first segment, 1 for second) are attached as an
#' attribute.
#'
#' @param seq_list Character vector or list; text to tokenize.
#' @param vocab A wordpiece vocabulary.
#' @param pad_to_length Integer; optional length to pad sequences to.
#' @param sep_token Character; token to use at end of each segment. Should exist
#'   in \code{vocab}.
#' @param cls_token Character; token to use at start of sequence. Should exist
#'   in \code{vocab}.
#' @param pad_token Character; token to use for padding sequence. Should exist
#'   in \code{vocab}.
#'
#' @return The tokenized output as a named integer vector, with token type ids
#'   attached as an attribute.
#' @export
#' @examples
#' vocab_path <- system.file("extdata", "tiny_vocab.txt", package = "wordpiece")
#' vocab <- wordpiece::load_vocab(vocab_file = vocab_path)
#' tokenize_input(c(
#'   "Here are some words.",
#'   "Here are some more words."
#' ),
#' vocab)
#' tokenize_input(list(
#'   c(
#'     "First sequence, first segment.",
#'     "First sequence, second segment."
#'   ),
#'   c(
#'     "Second sequence, first segment.",
#'     "Second sequence, second segment."
#'   )
#' ),
#' vocab)
tokenize_input <- function(seq_list,
                           vocab,
                           pad_to_length = NULL,
                           sep_token = "[SEP]",
                           cls_token = "[CLS]",
                           pad_token = "[PAD]") {
  if (any(purrr::map_int(seq_list, length) > 2)) {
    warning(
      "Examples must contain at most two distinct segments. ",
      "Segments beyond the second will be ignored."
    )
  }
  # throw warning if special tokens not in vocab. Maybe abort?
  if (!(sep_token %in% names(vocab))) {
    warning(sep_token, " is not in the specified vocabulary.")
  }
  if (!(cls_token %in% names(vocab))) {
    warning(cls_token, " is not in the specified vocabulary.")
  }
  if (!(pad_token %in% names(vocab))) {
    warning(pad_token, " is not in the specified vocabulary.")
  }
  sep_index <- vocab[[sep_token]]
  names(sep_index) <- sep_token
  cls_index <- vocab[[cls_token]]
  names(cls_index) <- cls_token
  if (!is.null(pad_to_length)) {
    pad_index <- vocab[[pad_token]]
    names(pad_index) <- pad_token
  }

  seq_nums <- seq_along(seq_list)
  to_return <- purrr::map(seq_nums, function(sn) {
    first_segment <- seq_list[[sn]][[1]]
    all_segments <- wordpiece::wordpiece_tokenize(text = first_segment,
                                                  vocab = vocab)
    second_segment <- NULL
    if (length(seq_list[[sn]]) > 1) {
      second_segment <- seq_list[[sn]][[2]]
      second_segment <- wordpiece::wordpiece_tokenize(text = second_segment,
                                                      vocab = vocab)
      all_segments <- c(all_segments, sep_index, second_segment)
    }
    inner_return <- c(cls_index, all_segments, sep_index)
    if (!is.null(pad_to_length)) {
      inner_return <- .pad_vector(inner_return,
                                  len = pad_to_length,
                                  padding = pad_index)
    }
    inner_return
  })

  tt_ids <- .get_token_type_ids(to_return, sep_index)

  # I don't particularly like doing it this way, but I can't think of a better
  # way at this time.
  # https://github.com/jonathanbratt/RBERT/issues/61
  attr(to_return, "tt_ids") <- tt_ids

  return(to_return)
}


# .get_token_type_ids ---------------------------------------------------------

#' Get Token Type IDs from Tokenization
#'
#' The token type is 0 for the first segement, and 1 for the second segment.
#' Segments are determined here by the SEP tokens.
#'
#' @param tokenization Output from \code{tokenize_input}.
#' @param sep_index Named integer value, as used within \code{tokenize_input}.
#'
#' @return Integer vector of token type ids.
#' @keywords internal
.get_token_type_ids <- function(tokenization, sep_index) {
  return(
    purrr::map(tokenization, function(seq) {
      # find the locations of all the SEP tokens
      seps <- which(seq == sep_index)
      if (length(seps) > 1) {
        # If more than one SEP, fill with zeros up to the first (inclusive),
        # and with ones after that.
        return(c(rep(0, seps[[1]]), rep(1, length(seq) - seps[[1]])))
      }
      # Otherwise, all zeros.
      return(rep(0, length(seq)))
    })
  )
}

