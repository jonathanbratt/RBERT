# reticulate:::conda_list()
# conda <- reticulate:::conda_binary("auto")
# system(paste(conda, "info --json"))

test_that("TF is working.", {
  # registry_versions <- reticulate::py_versions_windows()
  # anaconda_registry_versions <- subset(registry_versions,
  #                                      registry_versions$type == "Anaconda")
  # print(
  #   paste(
  #     file.path(anaconda_registry_versions$install_path, "Scripts", "conda.exe"),
  #     collapse = " | "
  #   )
  # )

  # condas <- paste(reticulate::conda_list()$name, collapse = " | ")
  # print(condas)
  # expect_identical(condas, "this is not that")
  # reticulate::use_condaenv('r-reticulate')
  expect_true(!is.null(tensorflow::tf_version()))
})
