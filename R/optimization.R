# Copyright 2019 Bedford Freeman & Worth Pub Grp LLC DBA Macmillan Learning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# create_optimizer --------------------------------------------------------

#' Create an optimizer training op
#'
#' \code{create_optimizer} doesn't actually return the optimizer object; it
#' returns the operation resulting from a tf.group() call.
#'
#' See also:
#'
#' \url{https://www.tensorflow.org/api_docs/python/tf/group}
#'
#' \url{https://stackoverflow.com/questions/41780655/what-is-the-difference-between-tf-group-and-tf-control-dependencies}
#'
#' The routine tf.gradients() is called in the course of this function.
#' \url{https://www.tensorflow.org/api_docs/python/tf/gradients}
#'
#' @param loss Float Tensor; the loss for this step (calculated elsewhere; in
#'   principle is a function of trainable parameter values).
#' @param init_lr Numeric; initial learning rate.
#' @param num_train_steps Integer; number of steps to train for.
#' @param num_warmup_steps Integer; number of steps to use for "warm-up".
#' @param use_tpu Logical; whether to use TPU.
#'
#' @return A training op: the result of a tensorflow group() of operations.
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'   reuse = tensorflow::tf$AUTO_REUSE
#' ), {
#'   totrain <- tensorflow::tf$get_variable(
#'     "totrain",
#'     tensorflow::shape(10L, 20L)
#'   )
#'   loss <- 2 * totrain
#'
#'   train_op <- create_optimizer(
#'     loss = loss,
#'     init_lr = 0.01,
#'     num_train_steps = 20L,
#'     num_warmup_steps = 10L,
#'     use_tpu = FALSE
#'   )
#' })
#' }
create_optimizer <- function(loss,
                             init_lr,
                             num_train_steps,
                             num_warmup_steps,
                             use_tpu) {
  # Return and create (if necessary) the global step tensor.
  # This is used for keeping track of total number of training steps.
  global_step <- tensorflow::tf$train$get_or_create_global_step()

  learning_rate <- tensorflow::tf$constant(
    value = init_lr,
    shape = list(),
    dtype = tensorflow::tf$float32
  )

  # Implements linear decay of the learning rate.
  # https://devdocs.io/tensorflow~python/tf/train/polynomial_decay
  learning_rate <- tensorflow::tf$train$polynomial_decay(
    learning_rate,
    global_step,
    num_train_steps,
    end_learning_rate = 0.0,
    power = 1.0,
    cycle = FALSE
  )

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if (num_warmup_steps > 0) {
    global_steps_int <- tensorflow::tf$cast(
      global_step,
      tensorflow::tf$int32
    )
    warmup_steps_int <- tensorflow::tf$constant(num_warmup_steps,
      dtype = tensorflow::tf$int32
    )

    global_steps_float <- tensorflow::tf$cast(
      global_steps_int,
      tensorflow::tf$float32
    )
    warmup_steps_float <- tensorflow::tf$cast(
      warmup_steps_int,
      tensorflow::tf$float32
    )

    warmup_percent_done <- global_steps_float / warmup_steps_float
    warmup_learning_rate <- init_lr * warmup_percent_done

    # This is casting a logical to a float...
    is_warmup <- tensorflow::tf$cast(
      global_steps_int < warmup_steps_int,
      tensorflow::tf$float32
    )
    # ...so that we can write an `if` statement like this? -JDB
    learning_rate <- (
      (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
  }

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer <- AdamWeightDecayOptimizer(
    learning_rate = learning_rate,
    weight_decay_rate = 0.01,
    beta_1 = 0.9,
    beta_2 = 0.999,
    epsilon = 1e-6,
    exclude_from_weight_decay = c("LayerNorm", "layer_norm", "bias")
  )

  if (use_tpu) {
    optimizer <- tensorflow::tf$contrib$tpu$CrossShardOptimizer(optimizer)
  }

  # This is a pretty important step. It's where the derivative of the loss
  # wrt all the variables is calculated. -JDB
  tvars <- tensorflow::tf$trainable_variables()
  grads <- tensorflow::tf$gradients(loss, tvars)

  # This is how the model was pre-trained.
  # https://devdocs.io/tensorflow~python/tf/clip_by_global_norm
  grads <- tensorflow::tf$clip_by_global_norm(grads, clip_norm = 1.0)[[1]]

  # In python, this was done with `zip(grads, tvars)`
  # Be sure this structure is compatible with downstream usage. -JDB
  grads_and_vars <- purrr::map2(grads, tvars, list)
  train_op <- optimizer$apply_gradients(grads_and_vars,
    global_step = global_step
  )

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this.  But if you use a
  # different optimizer, you should probably take this line out.
  new_global_step <- global_step + 1L
  train_op <- tensorflow::tf$group(
    train_op,
    list(global_step$assign(new_global_step))
  )
  return(train_op)
}


# class AdamWeightDecayOptimizer ------------------------------------------------


#' Constructor for objects of class AdamWeightDecayOptimizer
#'
#' A basic Adam optimizer that includes "correct" L2 weight decay.
#'
#' Inherits from class tf.train.Optimizer.
#' \url{https://devdocs.io/tensorflow~python/tf/train/optimizer}
#'
#' @param learning_rate Numeric Tensor (single element?); learning rate.
#' @param weight_decay_rate Numeric; weight decay rate.
#' @param beta_1 Numeric; parameter for Adam.
#' @param beta_2 Numeric; parameter for Adam.
#' @param epsilon Numeric; a tiny number to put a cap on update size by avoiding
#'   dividing by even smaller numbers.
#' @param exclude_from_weight_decay Character; list of parameter names to
#'   exclude from weight decay.
#' @param name Character; the name of the constructed object.
#'
#' @return An object of class "AdamWeightDecayOptimizer", which is a (hacky)
#'   modification of the tf.train.Optimizer class.
#' @export
#'
#' @examples
#' \dontrun{
#' with(tensorflow::tf$variable_scope("examples",
#'   reuse = tensorflow::tf$AUTO_REUSE
#' ), {
#'   optimizer <- AdamWeightDecayOptimizer(learning_rate = 0.01)
#' })
#' }
AdamWeightDecayOptimizer <- function(learning_rate,
                                     weight_decay_rate = 0.0,
                                     beta_1 = 0.9,
                                     beta_2 = 0.999,
                                     epsilon = 1e-6,
                                     exclude_from_weight_decay = NULL,
                                     name = "AdamWeightDecayOptimizer") {
  # python code uses `super` to call constructor for parent class. I'm not super
  # (hehe) familiar with this construction, so check this extra carefully. -JDB

  # Because this class is extending an existing tf class, keep its
  # structure compatible with the tf class.
  # Hmm, this is more subtle than I first realized. Objects of this class
  # will be passed back into tensorflow routines, so they need to meet the
  # class requirements. To replicate the python code, I want to extend the
  # python class ...in R. I could rebuild the child class in, say, R6,
  # but I don't know how to make this inherit from the parent class (there
  # may be a way in reticulate, but a Google search wasn't encouraging).
  # Another possibility (this is the direction I went):
  # The extended class really only replaces a single method: `apply_gradients`.
  # This method doesn't appear to change any of the object-owned variables, so
  # it doesn't really have be an object method (or even a class method); it
  # can just be a function that's defined in the same environment as its
  # "owning" object (so that it can access the appropriate parameter values).
  # This function will have the same semantics as proper object method, but
  # wouldn't be able to change object variables (which is OK, because it
  # won't try to). If any of the downstream code checks the class of the method,
  # then we'll be revealed as imposters, but otherwise this should work. -JDB

  this_opt <- tensorflow::tf$train$Optimizer(use_locking = FALSE, name = name)
  class(this_opt) <- c("AdamWeightDecayOptimizer", class(this_opt))

  # The variables below are good citizens of objects of the child class.
  # All of these will be "accessed" by the `apply_gradients` function defined
  # below, but not as object properties--they exist in the environment the
  # function was defined in. If these values are changed AFTER this point,
  # the function WILL NOT pick up the updated values. If it turns out we
  # need full object functionality, the fall-back plan is to do the class
  # extension in python, and `reticulate` it into R. -JDB
  this_opt$learning_rate <- learning_rate
  this_opt$weight_decay_rate <- weight_decay_rate
  this_opt$beta_1 <- beta_1
  this_opt$beta_2 <- beta_2
  this_opt$epsilon <- epsilon
  this_opt$exclude_from_weight_decay <- exclude_from_weight_decay

  # Here's where we sneakily replace the existing object method
  # with a function that's called the same way, but doesn't *actually*
  # have access to the class or the object.
  # For some reason, this approach reminds me of this fish's tongue:
  # https://en.wikipedia.org/wiki/Cymothoa_exigua   -JDB
  this_opt$apply_gradients <- function(grads_and_vars,
                                       global_step = NULL,
                                       name = NULL) {
    assignments <- list()
    for (gv in grads_and_vars) {
      grad <- gv[[1]]
      param <- gv[[2]]
      if (!is.null(grad) & !is.null(param)) {
        # In python, the regex below was wrapped in _get_variable_name.
        # (Get the variable name from the tensor name.) -JDB
        param_name <- param$name # tensor name
        match <- stringr::str_match(
          string = param_name,
          pattern = "^(.*):\\d+$"
        )[[2]]
        if (!is.na(match)) {
          param_name <- match # variable name
        }

        m <- tensorflow::tf$get_variable(
          name = paste0(param_name, "/adam_m"),
          shape = param$shape$as_list(),
          dtype = tensorflow::tf$float32,
          trainable = FALSE,
          initializer = tensorflow::tf$zeros_initializer()
        )
        v <- tensorflow::tf$get_variable(
          name = paste0(param_name, "/adam_v"),
          shape = param$shape$as_list(),
          dtype = tensorflow::tf$float32,
          trainable = FALSE,
          initializer = tensorflow::tf$zeros_initializer()
        )

        # Standard Adam update.
        # Note: here we're not accessing beta_1, etc. as object parameters, we
        # just know their original values because they're in the current
        # environment. THIS WOULDN'T WORK IF THESE VALUES HAD TO BE MUTABLE. But
        # I think they are effectively static in this case. -JDB
        next_m <- (tensorflow::tf$multiply(beta_1, m) +
          tensorflow::tf$multiply(1.0 - beta_1, grad))
        next_v <- (tensorflow::tf$multiply(beta_2, v) +
          tensorflow::tf$multiply(
            1.0 - beta_2,
            tensorflow::tf$square(grad)
          ))
        update <- next_m / (tensorflow::tf$sqrt(next_v) + epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange
        # ways.
        #
        # Instead we want to decay the weights in a manner that doesn't
        # interact with the m/v parameters. This is equivalent to adding the
        # square of the weights to the loss with plain (non-momentum) SGD.

        # In python, this bit was wrapped in _do_use_weight_decay.
        # Determine whether we want to do weight decay for these particular
        # variables. -JDB
        use_weight_decay <- TRUE
        if (weight_decay_rate == 0) {
          use_weight_decay <- FALSE
        }
        if (!is.null(exclude_from_weight_decay)) {
          for (r in exclude_from_weight_decay) {
            if (stringi::stri_detect(str = param_name, regex = r)) {
              use_weight_decay <- FALSE
            }
          }
        }

        if (use_weight_decay) {
          update <- update + weight_decay_rate * param
        }
        update_with_lr <- learning_rate * update

        next_param <- param - update_with_lr

        # Clumsy way to replicate python .extend method. Seems like there should
        # be a base function for this, but I can only find it in external
        # packages that I don't want to require just for this. -JDB
        assignments <- unlist(
          list(
            assignments,
            param$assign(next_param),
            m$assign(next_m),
            v$assign(next_v)
          ),
          recursive = FALSE
        )
      }
    }
    # In python, the list was "delisted" by `*assignments`. This seems to
    # be the best R alternative. I'm not entirely sure this is needed--the
    # tf.group function may be able to handle lists. -JDB
    return(
      do.call(function(...) {
        tensorflow::tf$group(..., name = name)
      },
      args = assignments
      )
    )
  } # end definition of `apply_gradients`

  return(this_opt)
}
