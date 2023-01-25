
<!-- README.md is generated from README.Rmd. Please edit that file -->

# RBERT <img src='man/figures/rbert_hex.png' align="right" height="138.5" />

<!-- badges: start -->

[![Lifecycle:
superseded](https://img.shields.io/badge/lifecycle-superseded-blue.svg)](https://lifecycle.r-lib.org/articles/stages.html#superseded)
[![Travis build
status](https://travis-ci.org/jonathanbratt/RBERT.svg?branch=master)](https://travis-ci.org/jonathanbratt/RBERT)
[![AppVeyor build
status](https://ci.appveyor.com/api/projects/status/github/jonathanbratt/RBERT?branch=master&svg=true)](https://ci.appveyor.com/project/jonathanbratt/RBERT)
[![Codecov test
coverage](https://codecov.io/gh/jonathanbratt/RBERT/branch/master/graph/badge.svg)](https://codecov.io/gh/jonathanbratt/RBERT?branch=master)
<!-- badges: end -->

We are re-implementing BERT for R in
[{torchtransformers}](https://github.com/macmillancontentscience/torchtransformers).
We find {torch} much easier to work with in R than {tensorflow}, and
strongly recommend starting there!

------------------------------------------------------------------------

RBERT is an R implementation of the Python package
[BERT](https://github.com/google-research/bert) developed at Google for
Natural Language Processing.

## Installation

You can install RBERT from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
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

The current CRAN version of reticulate (1.13) causes some issues with
the tensorflow installation. Rebooting your machine after installing
Anaconda seems to fix this issue, or upgrade to the development version
of reticulate.

``` r
devtools::install_github("rstudio/reticulate")
```

## Basic usage

RBERT is a work in progress. While fine-tuning a BERT model using RBERT
may be possible, it is not currently recommended.

RBERT is best suited for exploring pre-trained BERT models, and
obtaining contextual representations of input text for use as features
in downstream tasks.

-   See the “Introduction to RBERT” vignette included with the package
    for more specific examples.
-   For a quick explanation of what BERT is, see the “BERT Basics”
    vignette.
-   The package [RBERTviz](https://github.com/jonathanbratt/RBERTviz)
    provides tools for making fun and easy visualizations of BERT data.

## Running Tests

The first time you run the test suite, the 388.8MB bert_base_uncased.zip
file will download in your `tests/testthat/test_checkpoints` directory.
Subsequent test runs will use that download. This was our best
compromise to allow for relatively rapid testing without bloating the
repository.

## Disclaimer

This is not an officially supported Macmillan Learning product.

## Contact information

Questions or comments should be directed to Jonathan Bratt
(<jonathan.bratt@macmillan.com>) and Jon Harmon
(<jon.harmon@macmillan.com>).
