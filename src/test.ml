open Printf

module L = BatList

module Score_label = struct
  type t = bool * float (* (label, pred_score) *)
  let get_label (l, _) = l
  let get_score (_, s) = s
end

module ROC = MakeROC.Make(Score_label)

(* FBR: test reading data from sparse matrix *)

let main () =
  Log.set_log_level Log.DEBUG;
  Log.color_on ();
  let data_fn = "data/train_data.txt" in
  let sparse_data_fn = "data/train_data.csr" in
  let labels_fn = "data/train_labels.txt" in
  let preds =
    let params = Xgboost.default_gbtree_params () in
    let model =
      Xgboost.train
        ~debug:true
        Dense
        10
        params
        data_fn
        labels_fn in
    Xgboost.read_predictions
      (Xgboost.predict ~debug:true Dense model data_fn) in
  let sparse_preds =
    let params = Xgboost.default_gbtree_params () in
    let sparsity = Xgboost.Sparse 1831 in
    let model =
      Xgboost.train
        ~debug:true
        sparsity
        10
        params
        sparse_data_fn
        labels_fn in
    Xgboost.read_predictions
      (Xgboost.predict ~debug:true sparsity model sparse_data_fn) in
  let lin_preds =
    let params = Xgboost.default_linear_params () in
    let model =
      Xgboost.train
        ~debug:true
        Dense
        10
        params
        data_fn
        labels_fn in
    let preds_fn = Xgboost.predict ~debug:true Dense model data_fn in
    Xgboost.read_predictions preds_fn in
  assert(List.length preds = 88);
  assert(List.length sparse_preds = 88);
  assert(List.length lin_preds = 88);
  (* List.iter (printf "%f\n") predictions *)
  let labels =
    let labels_line = Utls.with_in_file labels_fn input_line in
    let label_strings = BatString.split_on_char '\t' labels_line in
    L.map (function
        | "1" -> true
        | "-1" -> false
        | other -> failwith other
      ) label_strings in
  let auc = ROC.auc (List.combine labels preds) in
  let sparse_auc = ROC.auc (List.combine labels sparse_preds) in
  let lin_auc = ROC.auc (List.combine labels lin_preds) in
  printf "AUC: %.3f\n" auc;
  printf "sparse AUC: %.3f\n" sparse_auc;
  printf "lin AUC: %.3f\n" lin_auc
  (* let sparse_lin_auc = ROC.auc (List.combine labels sparse_lin_preds) in
   * printf "sparse Lin AUC: %.3f\n" sparse_lin_auc; *)

let () = main ()
