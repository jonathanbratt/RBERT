
<!-- README.md is generated from README.Rmd. Please edit that file -->

# RBERT <img src='man/figures/rbert_hex.png' align="right" height="138.5" />

<!-- badges: start -->

[![Lifecycle:
maturing](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://www.tidyverse.org/lifecycle/#maturing)
[![Travis build
status](https://travis-ci.org/jonathanbratt/RBERT.svg?branch=master)](https://travis-ci.org/jonathanbratt/RBERT)
[![AppVeyor build
status](https://ci.appveyor.com/api/projects/status/github/jonathanbratt/RBERT?branch=master&svg=true)](https://ci.appveyor.com/project/jonathanbratt/RBERT)
[![Codecov test
coverage](https://codecov.io/gh/jonathanbratt/RBERT/branch/master/graph/badge.svg)](https://codecov.io/gh/jonathanbratt/RBERT?branch=master)
<!-- badges: end -->

RBERT is an R implementation of the Python package
[BERT](https://github.com/google-research/bert) developed at Google for
Natural Language Processing.

## Installation

You can install RBERT from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
# Set up a GITHUB_PAT with Sys.setenv(GITHUB_PAT = "YOURPATHERE")
devtools::install_github(
  "jonathanbratt/RBERT", 
  build_vignettes = TRUE
)
```

### TensorFlow Installation

RBERT requires TensorFlow. Currently the version must be \<= 1.13.1. You
can install it using the tensorflow package (installed as a dependency
of this package; see note below about Windows).

``` r
tensorflow::install_tensorflow(version = "1.13.1")
```

### Windows

The current CRAN version of reticulate (1.13) evidently causes some
issues with the tensorflow installation. Rebooting your machine after
installing Anaconda seems to fix this issue, or upgrade to the
development version of reticulate.

``` r
devtools::install_github("rstudio/reticulate")
```

## Basic usage

RBERT is a work in progress. While fine-tuning a BERT model using RBERT
may be possible, it is not currently recommended.

RBERT is best suited for exploring pre-trained BERT models, and
obtaining contextual representations of input text for use as features
in downstream tasks.

See the “Introduction to RBERT” vignette included with the package for
more specific examples.

## Disclaimer

This is not an officially supported Macmillan Learning product.

## Contact information

Questions or comments should be directed to Jonathan Bratt
(<jonathan.bratt@macmillan.com>) and Jon Harmon
(<jon.harmon@macmillan.com>).
