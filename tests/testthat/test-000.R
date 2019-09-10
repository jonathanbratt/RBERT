reticulate:::conda_list()
conda <- reticulate:::conda_binary("auto")
system(paste(conda, "info --json"))

test_that("TF is working.", {
  expect_true(!is.null(tensorflow::tf_version()))
})
