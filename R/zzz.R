# Copyright 2021 Bedford Freeman & Worth Pub Grp LLC DBA Macmillan Learning.
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
  .custom_layers[["position_embedding"]] <<-
    .make_custom_layer_position_embedding()
  .custom_layers[["bert_embeddings"]] <<- .make_custom_layer_bert_embeddings()
}
