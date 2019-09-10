# reticulate:::conda_list()
# conda <- reticulate:::conda_binary("auto")
# system(paste(conda, "info --json"))

test_that("TF is working.", {
  library(tensorflow)
  expect_true(!is.null(tensorflow::tf_version()))
})
