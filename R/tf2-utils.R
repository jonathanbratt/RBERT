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


# .update_list -------------------------------------------------------------


#' Update a List
#'
#' Updates the first list with matching named elements from the second list.
#' Returns the updated list. Elements in the second list that don't match
#' (names of) elements in the first list are ignored.
#'
#' @param list1 Named list to update.
#' @param list2 Named list to update with.
#'
#' @return The updated list.
#'
#' @keywords internal
.update_list <- function(list1, list2) {
  # For some reason, py to R conversion wasn't working here, so do manually:
  if (inherits(list1, "python.builtin.dict")) {
    lnames <- names(list1)
    list1 <- purrr::map(lnames, function(n) {list1[[n]]})
    names(list1) <- lnames
  }
  if (inherits(list2, "python.builtin.dict")) {
    lnames <- names(list2)
    list2 <- purrr::map(lnames, function(n) {list2[[n]]})
    names(list2) <- lnames
  }
  list1 <- as.list(list1)
  list2 <- as.list(list2)
  return(utils::modifyList(list1, list2, keep.null = TRUE)[names(list1)])
}

# .create_attention_mask ---------------------------------------------------

#' Create an Attention Mask
#'
#' Creates a 3D attention mask.
#'
#' Attention masks are used to control which words can pay attention to which.
#' This function takes an input mask with extent along the "to sequence" axis,
#' and broadcasts along the "from sequence" axis as well.
#'
#' @param from_shape List or integer vector. First two elements should be
#' \code{batch_size} and \code{from_seq_len}.
#' @param input_mask Tensor with dimensions \code{[batch_size, seq_len]}.
#'
#' @return A tensor of ones with dimensions \code{[batch_size, from_seq_len,
#'   seq_len]}.
#'
#' @keywords internal
.create_attention_mask <- function(from_shape, input_mask) {
  mask <- tensorflow::tf$cast(
    tensorflow::tf$expand_dims(input_mask, axis = 1L),
    tensorflow::tf$float32
  ) # [B, 1, T]

  ones <- tensorflow::tf$expand_dims(
    tensorflow::tf$ones(shape = from_shape[1:2],
                        dtype = tensorflow::tf$float32),
    axis = -1L
  )  # [B, F, 1]
  mask <- ones * mask  # broadcast

  return(mask)  # [B, F, T]
}


# .tf2 ------------------------------------------------------------

#' Check for TensorFlow 2
#'
#' Returns \code{TRUE} iff TensorFlow installation is version 2.x.
#'
#' @return \code{TRUE} iff TensorFlow installation is version 2.x.
#'
#' @keywords internal
.tf2 <- function() {
  return(grepl(pattern = "^2\\.",
               x = as.character(tensorflow::tf_version())))
}


# .pad_vector ---------------------------------------------------------

#' Pad a Vector to a Certain Length
#'
#' Pad or truncate the given vector to specified length.
#'
#' @param x Vector to pad.
#' @param len Integer; length to pad or truncate to.
#' @param padding Object to use for padding
#'
#' @return `x` padded or truncated to given length.
#' @keywords internal
.pad_vector <- function(x, len, padding) {
  # add check for len > 0; maybe start using assert?
  # https://github.com/jonathanbratt/RBERT/issues/60
  if(length(x) >= len) {
    return(x[1:len])
  }
  return(c(x, rep(padding, len-length(x))))
}

