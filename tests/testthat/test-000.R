# reticulate:::conda_list()
# conda <- reticulate:::conda_binary("auto")
# system(paste(conda, "info --json"))

paste(reticulate::conda_list()$name, collapse = " | ")

test_that("TF is working.", {
  expect_true(!is.null(tensorflow::tf_version()))
})
