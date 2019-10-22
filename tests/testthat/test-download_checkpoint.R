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


test_that("download_checkpoint works", {
  # checkpoint was downloaded in setup.R
  # Redownloading the checkpoint should occur without incident.
  new_cpdir <- download_BERT_checkpoint(model = "bert_base_uncased",
                                         dir = checkpoint_main_dir)
  expect_identical(new_cpdir, cpdir)

  testthat::expect_true(
    file.exists(file.path(cpdir, "vocab.txt")))
  testthat::expect_true(
    file.exists(file.path(cpdir, "bert_config.json")))
  testthat::expect_true(
    file.exists(file.path(cpdir, "bert_model.ckpt.index")))
  testthat::expect_true(
    file.exists(file.path(cpdir, "bert_model.ckpt.meta")))
  testthat::expect_true(
    file.exists(file.path(cpdir, "bert_model.ckpt.data-00000-of-00001")))
})

test_that("dir chooser works.", {
  expect_identical(
    .choose_BERT_dir("fake"), "fake"
  )
  temp_dir <- tempdir()
  test_dir <- paste0(temp_dir, "/testing")
  old_dir <- set_BERT_dir(test_dir)
  expect_identical(
    normalizePath(getOption("BERT.dir"), mustWork = FALSE),
    normalizePath(test_dir, mustWork = FALSE)
  )

  # If I don't send it a dir, first it should try the option.
  expect_identical(
    .choose_BERT_dir(NULL),
    normalizePath(test_dir, mustWork = FALSE)
  )

  # If I don't have an option or a dir, it should use the default.
  options(BERT.dir = NULL)
  default_dir <- rappdirs::user_cache_dir("RBERT")
  expect_identical(
    .choose_BERT_dir(NULL),
    default_dir
  )

  # Go back to the existing setting.
  options(BERT.dir = old_dir$BERT.dir)
})
