
type filename = string

type gbtree_params = { eta: float; (* learning rate *)
                       gamma: float; (* minimum loss reduction *)
                       max_depth: int; (* max depth of tree *)
                       min_child_weight: float; (* minimum sum of
                                                   instance weight *)
                       subsample: float; (* subsample ratio of
                                            training instances *)
                       colsample_bytree: float; (* subsample ratio of columns *)
                       num_parallel_tree: int } (* number of trees to grow
                                                   per round *)

type linear_params = { lambda: float; (* L2 regularization term on weights *)
                       lambda_bias: float; (* L2 regularization term on bias *)
                       alpha: float } (* L1 regularization term on weights *)

type booster =
  | Gbtree of gbtree_params
  | Gblinear of linear_params

val default_linear_params: unit -> booster

val default_gbtree_params: unit -> booster

type nb_columns = int
type sparsity = Dense
              | Sparse of nb_columns

val train: ?debug:bool ->
  sparsity -> int -> booster -> filename -> filename -> Result.t

val predict: ?debug:bool -> sparsity -> Result.t -> filename -> Result.t

val read_predictions: ?debug:bool -> Result.t -> float list
