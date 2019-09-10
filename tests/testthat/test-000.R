# reticulate:::conda_list()
# conda <- reticulate:::conda_binary("auto")
# system(paste(conda, "info --json"))

test_that("TF is working.", {
  tensorflow::install_tensorflow(version = "1.12.0", restart_session = FALSE)
  expect_true(!is.null(tensorflow::tf_version()))
})
