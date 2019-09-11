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


test_that("create_optimizer works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         totrain <- tensorflow::tf$get_variable("totrain",
                                                tensorflow::shape(10L, 20L))
         loss <- 2*totrain

         t_op <- create_optimizer(
           loss = loss,
           init_lr = 0.01,
           num_train_steps = 20L,
           num_warmup_steps = 10L,
           use_tpu = FALSE
         )
       })

  testthat::expect_is(t_op,
                      "tensorflow.python.framework.ops.Operation")

  testthat::expect_true(grepl(pattern = "group_deps", t_op$name))

  # now actually put some training variables in place...
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       totrain <- tensorflow::tf$get_variable("totrain",
                                             tensorflow::shape(10L, 20L))
  )

})


test_that("AdamWeightDecayOptimizer works", {
  with(tensorflow::tf$variable_scope("tests",
                                     reuse = tensorflow::tf$AUTO_REUSE),
       {
         awd_opt <- AdamWeightDecayOptimizer(learning_rate = 0.01)
       })

  testthat::expect_is(awd_opt,
                      "AdamWeightDecayOptimizer")
  testthat::expect_is(awd_opt,
                      "tensorflow.python.training.optimizer.Optimizer")
  # after our hack, `apply_gradients` is a function, not a method.
  testthat::expect_is(awd_opt$apply_gradients,
                      "python.builtin.function")
})

