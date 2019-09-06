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

  temp_directory <- tempdir()
  cpdir <- download_BERT_checkpoint(model = "bert_base_uncased",
                                    destination = temp_directory)
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

  unlink(temp_directory, recursive = TRUE)
})
