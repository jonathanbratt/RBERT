library(magrittr)
google_base_url <- "https://storage.googleapis.com/bert_models/"
scibert_base_url <- "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/"

checkpoint_url_map <- c(
  "bert_base_uncased" = paste0(
    google_base_url,
    "2018_10_18/uncased_L-12_H-768_A-12.zip"
  ),
  "bert_base_cased" = paste0(
    google_base_url,
    "2018_10_18/cased_L-12_H-768_A-12.zip"
  ),
  "bert_large_uncased" = paste0(
    google_base_url,
    "2018_10_18/uncased_L-24_H-1024_A-16.zip"
  ),
  "bert_large_cased" = paste0(
    google_base_url,
    "2018_10_18/cased_L-24_H-1024_A-16.zip"
  ),
  "bert_large_uncased_wwm" = paste0(
    google_base_url,
    "2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip"
  ),
  "bert_large_cased_wwm" = paste0(
    google_base_url,
    "2019_05_30/wwm_cased_L-24_H-1024_A-16.zip"
  ),
  "bert_base_multilingual_cased" = paste0(
    google_base_url,
    "2018_11_23/multi_cased_L-12_H-768_A-12.zip"
  ),
  "bert_base_chinese" = paste0(
    google_base_url,
    "2018_11_03/chinese_L-12_H-768_A-12.zip"
  ),
  "scibert_scivocab_uncased" = paste0(
    scibert_base_url,
    "scibert_scivocab_uncased.tar.gz"
  ),
  "scibert_scivocab_cased" = paste0(
    scibert_base_url,
    "scibert_scivocab_cased.tar.gz"
  ),
  "scibert_basevocab_uncased" = paste0(
    scibert_base_url,
    "scibert_basevocab_uncased.tar.gz"
  ),
  "scibert_basevocab_cased" = paste0(
    scibert_base_url,
    "scibert_basevocab_cased.tar.gz"
  )
)

# I want to convert this to a tibble with more info, but I don't want to
# reformat all that, so I'm using enframe.
checkpoint_url_map <- tibble::enframe(
  checkpoint_url_map, name = "model", value = "url"
) %>%
  dplyr::mutate(
    archive_type = c(
      rep("zip", 8),
      rep("tar-gzip", 4)
    )
  )

usethis::use_data(
  checkpoint_url_map,
  internal = TRUE,
  overwrite = TRUE
)
rm(
  google_base_url,
  scibert_base_url,
  checkpoint_url_map
)
