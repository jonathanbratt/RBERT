# new custom layer without wrapper --------------------------------------------------------

#' @export
custom_init_function <- function(param_list = list(name = "LayerNorm2"), ...) {
  # Quick way of mimicking the params-flow structure.
  # Basically, the object will look for relevant parameters in three places,
  # with increasing priority.
  #  1. The default values given below
  #  2. The parameters passed in via the param_list argument
  #  3. Any parameters explicitly named in the ... arguments
  # This is a convenient way to ensure that the needed parameters are
  # passed through the object structure.

  # Some params may not be used? Double check.

  self$params <- list(
    epsilon = 1e-12,
    initializer_range = 0.02,
    trainable = TRUE,
    name = "LayerNorm",
    dtype = tensorflow::tf$float32$name,
    dynamic = FALSE
  )
  # The initialization method takes a param_list argument, as well as any
  # other named arguments. These parameters are used to update the `params`
  # property.

  # stop("for some reason params is not converting back to an R list. May have to make a manual workaround.")
  # mydv <<- self$params
  # print(class(reticulate::py_to_r(self$params)))
  print(self$params)
  print(class(self$params))
  #
  self$params <- RBERT:::.update_list(self$params, param_list)

  print("after:")
  print(self$params)
  print(class(self$params))

  self$params <- RBERT:::.update_list(self$params, list(...))
  print("after again:")
  print(self$params)
  print(class(self$params))
  mydv <<- self$params



  self$gamma  <- NULL
  self$beta  <- NULL
  self$supports_masking <- TRUE

  super()$`__init__`()
}

#' @export
custom_build_function <- function(input_shape) {
  # `input_shape`: Integer vector; shape of tensor input to this layer.
  # Not sure if I need to set self$input_spec, or if that is optional.
  # I can't find the functions for creating `InputSpec` objects in
  # the R keras/tensorflow packages, nor is it done in the examples.
  # So I will just skip this part for now. -JDB
  # self$input_spec <- (something like) keras::InputSpec(shape=input_shape)

  self$gamma <- self$add_weight(name = "gamma",
                                shape = input_shape[[length(input_shape)]],
                                initializer = keras::initializer_ones(),
                                trainable = TRUE)

  self$beta <- self$add_weight(name = "beta",
                               shape = input_shape[[length(input_shape)]],
                               initializer = keras::initializer_zeros(),
                               trainable = TRUE)

  super()$build(input_shape)
}


#' @export
custom_call_function <- function(inputs, ...) {
  # double check the axes indexing conventions here.
  mv <- tensorflow::tf$nn$moments(inputs, axes = -1L, keepdims = TRUE)
  mean <- mv[[1]]
  var <- mv[[2]] # uses N, not N-1, in denominator. Just FYI.

  inv <- self$gamma*tensorflow::tf$math$rsqrt(var + self$params$epsilon)

  res <- inputs*tensorflow::tf$cast(inv, inputs$dtype) +
    tensorflow::tf$cast(self$beta - mean*inv, inputs$dtype)

  return(res)
}

#' @export
make_custom_keras_layer_object <- function() {
  layer_function <- keras::Layer(
    # my_layer(
    classname = "LayerTestX",
    initialize = custom_init_function,


    build = custom_build_function,

    call = custom_call_function,

    compute_output_shape = function(input_shape) {
      return(input_shape)
    },

    compute_mask = function(inputs, mask = NULL) {
      return(mask)
    }

  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' @export
layer_custom_wrapper <- function(object,
                                    name = NULL,
                                    trainable = NULL,
                                    param_list = list(),
                                    ...) {
  keras::create_layer(layer_class = .custom_layers$test_x,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        param_list = param_list,
                        ...
                      )
                      )
}



# custom dense layer for testing --------------------------------------------------------

#' @keywords internal
custom_init_function_dense <- function(num_outputs, ...) {
  self$num_outputs <- num_outputs

  super()$`__init__`()
}

#' @keywords internal
custom_build_function_dense <- function(input_shape) {
  self$kernel <- self$add_weight(
    name = 'kernel',
    shape = list(input_shape[[2]], self$num_outputs)
  )

  super()$build(input_shape)
}


#' @keywords internal
custom_call_function_dense <- function(inputs, ...) {
  res <- tensorflow::tf$matmul(inputs, self$kernel)
  return(res)
}

#' @keywords internal
make_custom_dense_layer_object <- function() {
  layer_function <- keras::Layer(
    # my_layer(
    classname = "CustomDense",
    num_outputs = NULL,
    kernel = NULL,
    initialize = custom_init_function_dense,
    build = custom_build_function_dense,
    call = custom_call_function_dense,

    compute_output_shape = function(input_shape) {
      return(self$num_outputs)
    },

    compute_mask = function(inputs, mask = NULL) {
      return(mask)
    }

  )

  python_layer_object <- attr(layer_function, which = "layer")
  return(python_layer_object)
}


#' @export
layer_dense_custom_wrapper <- function(object,
                                 name = NULL,
                                 trainable = NULL,
                                 num_outputs,
                                 ...) {
  keras::create_layer(layer_class = .custom_layers$dense,
                      object = object,
                      args = list(
                        name = name,
                        trainable = trainable,
                        num_outputs = as.integer(num_outputs),
                        ...
                      )
  )
}




# onload ------------------------------------------------------------------


.custom_layers <- list()

.onLoad <-  function(libname, pkgname) {
  .custom_layers[["transformer_encoder_single"]] <<-
    .make_custom_layer_transformer_encoder_single()
  .custom_layers[["transformer_encoder"]] <<-
    .make_custom_layer_transformer_encoder()
  .custom_layers[["BERT"]] <<- .make_custom_layer_BERT()
  .custom_layers[["attention"]] <<- .make_custom_layer_attention()
  .custom_layers[["layernorm"]] <<- .make_custom_layer_layernorm()
  .custom_layers[["proj_add_norm"]] <<- .make_custom_layer_proj_add_norm()
  .custom_layers[["position_embedding"]] <<- .make_custom_layer_position_embedding()
  .custom_layers[["bert_embeddings"]] <<- .make_custom_layer_bert_embeddings()
  # .custom_layers[["dense"]] <<-  make_custom_dense_layer_object()
}
