
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
