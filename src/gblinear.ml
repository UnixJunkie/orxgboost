
open Printf

(* gradient boosted regressor using Gblinear *)

module L = BatList
module Log = Dolog.Log

type filename = string

type params = { eta: float; (* learning rate in [0.0:1.0] *)
                lambda: float; (* L2 regularization term on weights *)
                alpha: float; (* L1 regularization term on weights *)
                nrounds: int (* number of training rounds *) }

(* train model and return the filename it was saved to upon success *)
let train
    ?debug:(debug = false)
    (params: params)
    (data_fn: filename): Result.t =
  let model_fn: filename = Filename.temp_file "orxgboost_model_" ".bin" in
  (* create R script and store it in a temp file *)
  let r_script_fn = Filename.temp_file "orxgboost_train_" ".r" in
  Utls.with_out_file r_script_fn (fun out ->
      fprintf out
        "library(xgboost)\n\
         training_set <- as.matrix(read.table('%s', colClasses = 'numeric',\n\
                                   header = TRUE))\n\
         cols_count = dim(training_set)[2]\n\
         x <- training_set[, 2:cols_count]\n\
         y <- training_set[, 1:1]\n\
         stopifnot(cols_count == length(y))\n\
         gbtree <- xgboost(data = x, label = y, booster='gblinear', eta = %f,\n\
                           objective = 'reg:squarederror',\n\
                           eval_metric = 'rmse',\n\
                           nrounds = %d, lambda = %f, alpha = %f)\n\
         xgb.save(gbtree, '%s')\n\
         quit()\n"
        data_fn params.eta params.nrounds params.lambda params.alpha model_fn
    );
  let r_log_fn = Filename.temp_file "orxgboost_train_" ".log" in
  (* execute it *)
  let cmd = sprintf "R --vanilla --slave < %s 2>&1 > %s" r_script_fn r_log_fn in
  if debug then Log.debug "%s" cmd;
  if Sys.command cmd <> 0 then
    Utls.collect_script_and_log debug r_script_fn r_log_fn model_fn
  else
    Utls.ignore_fst
      (if not debug then L.iter Sys.remove [r_script_fn; r_log_fn])
      (Result.Ok model_fn)

(* use model in 'model_fn' to predict decision values for test data in 'data_fn'
   and return the filename containing values upon success *)
let predict ?debug:(debug = false) (maybe_model_fn: Result.t) (data_fn: filename)
  : float list =
  match maybe_model_fn with
  | Error err -> failwith ("Gblinear.predict: model error: " ^ err)
  | Ok model_fn ->
    let predictions_fn = Filename.temp_file "orxgboost_predictions_" ".txt" in
    (* create R script in temp file *)
    let r_script_fn = Filename.temp_file "orxgboost_predict_" ".r" in
    Utls.with_out_file r_script_fn (fun out ->
        fprintf out
          "library(xgboost)\n\
           test_set <- as.matrix(read.table('%s', colClasses = 'numeric',\n\
                                            header = TRUE))\n\
           lines_count = dim(test_set)[1]\n\
           cols_count = dim(test_set)[2]\n\
           x <- test_set[, 2:cols_count]\n\
           xgb.load('%s')\n\
           values <- predict(gbtree, x)\n\
           stopifnot(lines_count == length(values))\n\
           write.table(values, file = '%s', sep = '\n',\n\
                       row.names = F, col.names = F)\n\
           quit()\n"
          data_fn model_fn predictions_fn
      );
    (* execute it *)
    let r_log_fn = Filename.temp_file "orxgboost_predict_" ".log" in
    let cmd = sprintf "R --vanilla --slave < %s 2>&1 > %s" r_script_fn r_log_fn in
    if debug then Log.debug "%s" cmd;
    if Sys.command cmd <> 0 then
      match Utls.collect_script_and_log
              debug r_script_fn r_log_fn predictions_fn with
      | Ok _ -> assert(false)
      | Error err -> failwith ("Gblinear.predict: R error: " ^ err)
    else
      let preds = Utls.read_predictions debug (Result.Ok predictions_fn) in
      (if not debug then
         L.iter Sys.remove [r_script_fn; r_log_fn; predictions_fn]);
      preds
