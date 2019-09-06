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


# convert_to_unicode ------------------------------------------------------


#' Convert `text` to Unicode
#'
#' See documentation for `Encoding` for more information.
#' Assumes utf-8 input.
#'
#' @param text character scalar to convert to unicode
#'
#' @return input text, converted to unicode if applicable
#' @export
#'
#' @examples
#' convert_to_unicode("fa\xC3\xA7ile")
convert_to_unicode <- function(text) {
  if (validUTF8(text)) { # this seems to work for utf-8 and 'bytes' encodings
    Encoding(text) <- "UTF-8"
    return(text)
  } else {
    stop("Unsupported string type.")
  }
}


# load_vocab --------------------------------------------------------------


#' Load a vocabulary file
#'
#' @param vocab_file path to vocabulary file. File is assumed to be a text
#' file, with one token per line, with the line number corresponding to the
#' index of that token in the vocabulary.
#'
#' @return In the BERT Python code, the vocab is returned as an OrderedDict
#' from the collections package. Here we return the vocab as a named integer
#' vector. Names are tokens in vocabulary, values are integer indices.
#'
#' @export
#'
#' @examples
#' \dontrun{ vocab <- load_vocab(vocab_file = "vocab.txt") }
load_vocab <- function(vocab_file) {
  token_list <- readLines(vocab_file)
  if (length(token_list)==0) {
    return(integer(0))
  }
  token_list <- purrr::map(token_list, function(token) {
    convert_to_unicode(trimws(token))})
  index_list <- seq_along(token_list) - 1 # vocab is zero-indexed
  names(index_list) <- token_list
  return(index_list)
}


# convert_by_vocab --------------------------------------------------------


#' Convert a sequence of tokens/ids using the provided vocab.
#'
#' @param vocab Vocabulary; provides mapping from index to tokens. (This may
#' be in fact an "inverse vocabulary", where the names are the indices and
#' the values are the tokens.)
#' @param items Vector of the keys (names in the vocab vector) to "convert".
#'
#' @return Vector of the values in `vocab` corresponding to `items`.
#' (The names on the returned vector are kept.)
#'
#' @export
#'
#' @examples
#' convert_by_vocab(c("token1"=0, "token2"=1), "token1")
convert_by_vocab <- function(vocab, items) {
  return(vocab[items])
}

#' @describeIn convert_by_vocab Wrapper function for specifically converting
#' tokens to ids.
#'
#' @param tokens Equivalent to items.
#' @export
convert_tokens_to_ids <- function(vocab, tokens) {
  return(convert_by_vocab(vocab, tokens))
}

#' @describeIn convert_by_vocab Wrapper function for specifically converting
#' ids to tokens.
#'
#' @param inv_vocab Equivalent to vocab.
#' @param ids Equivalent to items.
#'
#' @export
convert_ids_to_tokens <- function(inv_vocab, ids) {
  return(convert_by_vocab(inv_vocab, ids))
}


# whitespace_tokenize -----------------------------------------------------


#' Run basic whitespace cleaning and splitting on a piece of text.
#'
#' @param text Character scalar to tokenize.
#'
#' @return Character vector of tokens.
#' @export
#'
#' @examples
#' whitespace_tokenize(text = " some\ttext \n with  whitespace ")
whitespace_tokenize <- function(text) {
  return(
    unlist(
      stringi::stri_split_regex(text, "\\s", omit_empty = TRUE)
    )
  )
}


# class FullTokenizer -----------------------------------------------------


#' Construct objects of FullTokenizer class.
#'
#' @param vocab_file Path to text file containing list of vocabulary tokens.
#' @param do_lower_case Logical: do we convert everything to lowercase?
#'
#' @return An object of class FullTokenizer.
#' @export
#'
#' @examples
#' \dontrun{f_tokenizer <- FullTokenizer("vocab.txt", TRUE) }
FullTokenizer <- function(vocab_file, do_lower_case = TRUE) {
  vocab <- load_vocab(vocab_file)
  inv_vocab <- names(vocab)
  names(inv_vocab) <- vocab
  basic_tokenizer <- BasicTokenizer(do_lower_case = do_lower_case)
  wordpiece_tokenizer <- WordpieceTokenizer(vocab = vocab)
  obj <- list("do_lower_case" = do_lower_case,
              "vocab" = vocab,
              "inv_vocab" = inv_vocab,
              "basic_tokenizer" = basic_tokenizer,
              "wordpiece_tokenizer" = wordpiece_tokenizer)
  class(obj) <- "FullTokenizer"
  return(obj)
}



# generic tokenize --------------------------------------------------------


#' Tokenizers for various objects.
#'
#' This tokenizer performs some basic cleaning, then splits up text on
#' whitespace and punctuation.
#'
#' @param tokenizer The Tokenizer object to refer to.
#' @param text The text to tokenize. For tokenize.WordpieceTokenizer, the text
#'   should have already been passed through BasicTokenizer.
#'
#' @return A list of tokens.
#' @export
tokenize <- function (tokenizer, text) {
  UseMethod("tokenize", tokenizer)
}


# tokenize.FullTokenizer --------------------------------------------------


#' @describeIn tokenize Tokenizer method for objects of FullTokenizer class.
#' @export
#'
#' @examples
#' \dontrun{
#' tokenizer <- FullTokenizer("vocab.txt", TRUE)
#' tokenize(tokenizer, text = "a bunch of words")
#' }
tokenize.FullTokenizer <- function(tokenizer, text) {
  b_tokens <- tokenize(tokenizer$basic_tokenizer, text) # this is really ugly.

  # We can't use purrr::map_chr here, since the output of .f is itself a vector
  # of variable length (map_chr died trying...). Use map + unlist.
  split_tokens <- purrr::map(b_tokens,
                             function(bt) {
                               tokenize(tokenizer$wordpiece_tokenizer, bt)
                             })
  return(unlist(split_tokens))
}



# class BasicTokenizer ----------------------------------------------------


#' Construct objects of BasicTokenizer class.
#'
#' (I'm not sure that this object-based approach is best for R implementation,
#' but for now just trying to reproduce python functionality.)
#'
#' Has methods: `tokenize.BasicTokenizer()` `run_strip_accents.BasicTokenizer()`
#' (internal use) `run_split_on_punc.BasicTokenizer()` (internal use)
#' `tokenize_chinese_chars.BasicTokenizer()` (internal use)
#' `is_chinese_char.BasicTokenizer()` (internal use)
#' `clean_text.BasicTokenizer()` (internal use)
#'
#' @param do_lower_case Logical; the value to give to the "do_lower_case"
#'   argument in the BasicTokenizer object.
#'
#' @return an object of class BasicTokenizer
#' @export
#'
#' @examples
#' \dontrun{
#' b_tokenizer <- BasicTokenizer(TRUE)
#' }
BasicTokenizer <- function(do_lower_case = TRUE) {
  obj <- list("do_lower_case" = do_lower_case)
  class(obj) <- "BasicTokenizer"
  return(obj)
}


# strip_accents -----------------------------------------------------------


#' Strip accents from a piece of text.
#'
#' (R implementation of BasicTokenizer._run_strip_accents from
#' BERT: tokenization.py.)
#'
#' @param text A character scalar, encoded as utf-8.
#'
#' @return text with accents removed.
#'
#' @keywords internal
strip_accents <- function(text) {
  # perhaps confirm utf-8 first?
  text <- stringi::stri_trans_nfd(text)

  return(
    apply_to_chars(text,
                   function(char) {
                     if(stringi::stri_detect_charclass(char, "\\p{Mn}")) {
                       return("")
                     }
                     return(char)
                   })
  )
}


# split_on_punc -----------------------------------------------------------


#' Split text on punctuation.
#'
#' (R implementation of BasicTokenizer._run_split_on_punc from
#' BERT: tokenization.py.)
#'
#' @param text A character scalar, encoded as utf-8.
#'
#' @return The input text as a character vector, split on punctuation
#' characters.
#'
#' @keywords internal
split_on_punc <- function(text) {
  # this feels icky, but try to break it :-P
  # Put a unique marker around each punctuation char, then split on the
  # marker (since we want the punctuation to be included in split).
  sep_marker <- "a!b"
  output <- apply_to_chars(text,
                           function(char) {
                             if(is_punctuation(char)) {
                               return(paste0(sep_marker, char, sep_marker))
                             }
                             return(char)
                           })
  return(
    unlist(
      stringi::stri_split_fixed(output, sep_marker, omit_empty = TRUE)
    )
  )
}


# tokenize_chinese_chars --------------------------------------------------


#' Add whitespace around any CJK character.
#'
#' (R implementation of BasicTokenizer._tokenize_chinese_chars from
#' BERT: tokenization.py.) This may result in doubled-up spaces,
#' but that's the behavior of the python code...
#'
#' @param text A character scalar.
#'
#' @return Text with spaces around CJK characters.
#'
#' @keywords internal
tokenize_chinese_chars <- function(text) {
  return(
    apply_to_chars(text,
                   function(char) {
                     cp <- utf8ToInt(char)
                     if(is_chinese_char(cp)) {
                       return(paste0(" ", char, " "))
                     }
                     return(char)
                   })
  )
}


# is_chinese_char ---------------------------------------------------------


#' Check whether cp is the codepoint of a CJK character.
#'
#' (R implementation of BasicTokenizer._is_chinese_char from
#' BERT: tokenization.py. From that file:
#'  This defines a "chinese character" as anything in the CJK Unicode block:
#'   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
#'
#' Note that the CJK Unicode block is NOT all Japanese and Korean characters,
#' despite its name. The modern Korean Hangul alphabet is a different block,
#' as is Japanese Hiragana and Katakana. Those alphabets are used to write
#' space-separated words, so they are not treated specially and are handled
#' like the alphabets of the other languages.)
#'
#' @param cp A unicode codepoint, as an integer.
#'
#' @return Logical TRUE if cp is codepoint of a CJK character.
#'
#' @keywords internal
is_chinese_char <- function(cp) {
  if ((cp >= 0x4E00 & cp <= 0x9FFF) |
      (cp >= 0x3400 & cp <= 0x4DBF) |
      (cp >= 0x20000 & cp <= 0x2A6DF) |
      (cp >= 0x2A700 & cp <= 0x2B73F) |
      (cp >= 0x2B740 & cp <= 0x2B81F) |
      (cp >= 0x2B820 & cp <= 0x2CEAF) |
      (cp >= 0xF900 & cp <= 0xFAFF) |
      (cp >= 0x2F800 & cp <= 0x2FA1F)) {
    return(TRUE)
  }
  return(FALSE)
}


# clean_text --------------------------------------------------------------


#' Perform invalid character removal and whitespace cleanup on text.
#'
#' (R implementation of BasicTokenizer._clean_text from
#' BERT: tokenization.py.)
#'
#' @param text A character scalar.
#'
#' @return Cleaned up text.
#'
#' @keywords internal
clean_text <- function(text) {
  return(
    apply_to_chars(text,
                   function(char) {
                     cp <- utf8ToInt(char)
                     if (cp == 0 | cp == 0xfffd | is_control(char)) {
                       return("")
                     } else if (is_whitespace(char)) {
                       return(" ")
                     }
                     return(char)
                   })
  )
}


# tokenize.BasicTokenizer -------------------------------------------------

#' @describeIn tokenize Tokenizer method for objects of BasicTokenizer class.
#' @export
#'
#' @examples
#' \dontrun{
#' tokenizer <- BasicTokenizer(TRUE)
#' tokenize(tokenizer, text = "a bunch of words")
#' }
tokenize.BasicTokenizer <- function(tokenizer, text) {
  text <- convert_to_unicode(text)
  text <- clean_text(text)

  # From BERT: tokenization.py:
  # This was added on November 1st, 2018 for the multilingual and Chinese
  # models. This is also applied to the English models now, but it doesn't
  # matter since the English models were not trained on any Chinese data
  # and generally don't have any Chinese data in them (there are Chinese
  # characters in the vocabulary because Wikipedia does have some Chinese
  # words in the English Wikipedia.).
  text  <- tokenize_chinese_chars(text)

  # We can't use purrr::map_chr here, since the output of .f is itself a vector
  # of variable length (map_chr died trying...). Use map + unlist.
  output_tokens <- purrr::map(whitespace_tokenize(text),
                              .f = function(orig_token, do_lower_case) {
                                if (do_lower_case) {
                                  orig_token <- tolower(orig_token)
                                }
                                orig_token <- strip_accents(orig_token)
                                return(split_on_punc(orig_token))
                              },
                              do_lower_case = tokenizer$do_lower_case
                              )
  return(unlist(output_tokens))

}


# class WordpieceTokenizer ------------------------------------------------


#' Construct objects of WordpieceTokenizer class.
#'
#' (I'm not sure that this object-based approach is best for R implementation,
#' but for now just trying to reproduce python functionality.)
#'
#' Has method: tokenize.WordpieceTokenizer()
#'
#' @param vocab Recognized vocabulary tokens, as a named integer vector. (Name
#' is token, value is index.)
#' @param unk_token Token to use for unknown words.
#' @param max_input_chars_per_word Length of longest word we will recognize.
#'
#' @return an object of class WordpieceTokenizer
#' @export
#'
#' @examples
#' \dontrun{
#' vocab <- load_vocab(vocab_file = "vocab.txt")
#' wp_tokenizer <- WordpieceTokenizer(vocab)
#' }
WordpieceTokenizer <- function(vocab,
                               unk_token = "[UNK]",
                               max_input_chars_per_word = 200) {
  obj <- list("vocab" = vocab,
              "unk_token" = unk_token,
              "max_input_chars_per_word" = max_input_chars_per_word)
  class(obj) <- "WordpieceTokenizer"
  return(obj)
}


# tokenize.WordpieceTokenizer ---------------------------------------------



#' @describeIn tokenize Tokenizer method for objects of WordpieceTokenizer
#'   class. This uses a greedy longest-match-first algorithm to perform
#'   tokenization using the given vocabulary. For example: input = "unaffable"
#'   output = list("un", "##aff", "##able") ... although, ironically, the BERT
#'   vocabulary actually gives output = list("una", "##ffa", "##ble") for this
#'   example, even though they use it as an example in their code.
#' @export
#'
#' @examples
#' \dontrun{
#' vocab <- load_vocab(vocab_file = "vocab.txt")
#' tokenizer <- WordpieceTokenizer(vocab)
#' tokenize(tokenizer, text = "a bunch of words")
#' }
tokenize.WordpieceTokenizer <- function(tokenizer, text) {
  text <- convert_to_unicode(text)
  # departing from structure of python code for efficiency
  # We can't use purrr::map_chr here, since the output of .f is itself a vector
  # of variable length (map_chr died trying...). Use map + unlist.
  output_tokens <- purrr::map(whitespace_tokenize(text),
                              .f = tokenize_word,
                              vocab = tokenizer$vocab,
                              unk_token = tokenizer$unk_token,
                              max_chars = tokenizer$max_input_chars_per_word)
  return(unlist(output_tokens))
}


# tokenize_word -----------------------------------------------------------


#' Tokenize a single "word" (no whitespace).
#'
#' In BERT: tokenization.py,
#' this code is inside the tokenize method for WordpieceTokenizer objects.
#' I've moved it into its own function for clarity.
#' Punctuation should already have been removed from the word.
#'
#' @param word Word to tokenize.
#' @param vocab Character vector containing vocabulary words
#' @param unk_token Token to represent unknown words.
#' @param max_chars Maximum length of word recognized.
#'
#' @return Input word as a list of tokens.
#' @export
#'
#' @examples
#' tokenize_word("unknown", vocab = c("un" = 0, "##known" = 1))
#' tokenize_word("known", vocab = c("un" = 0, "##known" = 1))
tokenize_word <- function(word, vocab, unk_token = "[UNK]", max_chars = 100) {
  vocab <- names(vocab) # quick patch for now.
  if (stringi::stri_length(word) > max_chars) {
    return(unk_token)
  }
  if (word %in% vocab) {
    return(word)
  }

  is_bad  <- FALSE
  start <- 1
  sub_tokens <- character(0)
  while (start <= stringi::stri_length(word)) {
    end <- stringi::stri_length(word)

    cur_substr  <- NA_character_
    while (start <= end) {
      sub_str <- substr(word, start, end)   # inclusive on both ends
      if (start > 1) {  # means this substring is a suffix, so add '##'
        sub_str <- paste0("##", sub_str)
      }
      if (sub_str %in% vocab) {
        cur_substr <- sub_str
        break
      }
      end <- end - 1
    }
    if (is.na(cur_substr) ) {
      is_bad <-  TRUE
      break
    }

    sub_tokens <- append(sub_tokens, cur_substr)
    start <- end + 1 # pick up where we left off
  }

  if (is_bad) {
    return(unk_token)
  }
  return(sub_tokens)
}


# is_whitespace -----------------------------------------------------------


#' Check whether `char` is a whitespace character.
#'
#' (R implementation of _is_whitespace from BERT: tokenization.py.)
#'
#' "\\t", "\\n", and "\\r" are technically control characters but we treat them
#' as whitespace since they are generally considered as such.
#'
#' @param char A character scalar, comprising a single unicode character.
#'
#' @return TRUE if char is a whitespace character.
#'
#' @keywords internal
is_whitespace <- function(char) {
  # This is a way to check the unicode general category:
  # stringi::stri_detect_charclass(char, "\\p{Zs}")

  if (char %in% c(" ", "\t", "\n", "\r")) {
    return(TRUE)
  }
  return(stringi::stri_detect_charclass(char, "\\p{Zs}"))
}

# is_control --------------------------------------------------------------


#' Check whether `char` is a control character.
#'
#' (R implementation of _is_control from BERT: tokenization.py.)
#'
#' "\\t", "\\n", and "\\r" are technically control characters but we treat them
#' as whitespace since they are generally considered as such.
#' @param char A character scalar, comprising a single unicode character.
#'
#' @return TRUE if char is a control character.
#'
#' @keywords internal
is_control <- function(char) {
  if (char %in% c(" ", "\t", "\n", "\r")) {
    return(FALSE)
  }
  return(stringi::stri_detect_charclass(char, "\\p{C}"))
}

# is_punctuation ----------------------------------------------------------


#' Check whether `char` is a punctuation character.
#'
#' (R implementation of _is_punctuation from BERT: tokenization.py.)
#'
#' We treat all non-letter/number ASCII as punctuation.
#' Characters such as "^", "$", and "`" are not in the Unicode
#' Punctuation class but we treat them as punctuation anyway, for
#' consistency.
#' @param char A character scalar, comprising a single unicode character.
#'
#' @return TRUE if char is a punctuation character.
#'
#' @keywords internal
is_punctuation <- function(char) {
  cp <- utf8ToInt(char)
  if ((cp >= 33 & cp <= 47) | (cp >= 58 & cp <= 64) |
      (cp >= 91 & cp <= 96) | (cp >= 123 & cp <= 126)) {
    return(TRUE)
  }
  return(stringi::stri_detect_charclass(char, "\\p{P}"))
}


# apply_to_chars ----------------------------------------------------------


#' Apply a function to each character in a string.
#'
#' Utility function for something done a lot in this package.
#'
#' @param text A character scalar to process.
#' @param .f The function to apply to each character. Should return a character
#' scalar, given a single-character input.
#' @param ... Other arguments to pass to .f.
#'
#' @return The character scalar obtained by applying the given function to
#' each character of the input string, and concatenating the results.
#'
#' @keywords internal
apply_to_chars <- function(text, .f, ...) {
  paste(
    purrr::map_chr(unlist(strsplit(text, "")), .f, ...),
    collapse = ""
  )
}
