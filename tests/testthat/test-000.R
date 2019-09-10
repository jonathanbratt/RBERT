# reticulate:::conda_list()
# conda <- reticulate:::conda_binary("auto")
# system(paste(conda, "info --json"))

test_that("TF is working.", {
  condas <- paste(reticulate::conda_list()$name, collapse = " | ")
  print(condas)
  expect_identical(condas, "this is not that")
  expect_true(!is.null(tensorflow::tf_version()))
})
