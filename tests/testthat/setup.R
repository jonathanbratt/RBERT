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

checkpoint_main_dir <- tempdir()

# We need the checkpoint to be available for the other tests, so "download" it
# here. We use a mock function for the part that does the actual downloading,
# and instead copy from tests/testthat/test_checkpoints.
dont_download_checkpoint <- function(url, checkpoint_zip_path) {
  root_dir <- "test_checkpoints"

  from_file <- switch(
    url,
    "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip" = "bert_base_uncased.zip",
    "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz" = "test_checkpoint.tar.gz"
  )

  from_path <- file.path(root_dir, from_file)

  file.copy(
    from = from_path,
    to = checkpoint_zip_path,
    overwrite = TRUE
  )

  invisible(TRUE)
}

mockery::stub(
  where = download_BERT_checkpoint,
  what = ".download_BERT_checkpoint",
  how = dont_download_checkpoint
)

print("Setting up test checkpoint.")
cpdir <- download_BERT_checkpoint(model = "bert_base_uncased",
                                  dir = checkpoint_main_dir)
