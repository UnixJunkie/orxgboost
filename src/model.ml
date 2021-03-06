
module CLI = Minicli.CLI
module Gblinear = Orxgboost.Gblinear
module Fn = Filename
module Gnuplot = Orxgboost.Gnuplot
module L = BatList
module Utls = Orxgboost.Utls
module Log = Dolog.Log
module Parmap = Parany.Parmap
module Result = Orxgboost.Result

open Printf

let extract_values verbose fn =
  let actual_fn = Fn.temp_file "orxgboost_test_" ".txt" in
  (* NR > 1: skip CSV header line *)
  let cmd = sprintf "awk '(NR > 1){print $1}' %s > %s" fn actual_fn in
  Utls.run_command verbose cmd;
  let actual = Utls.float_list_of_file actual_fn in
  (* filesystem cleanup *)
  (if not verbose then Sys.remove actual_fn);
  actual

let train_test_dump csv_header train test =
  let train_fn = Fn.temp_file "orxgboost_train_" ".csv" in
  let test_fn = Fn.temp_file "orxgboost_test_" ".csv" in
  Utls.lines_to_file train_fn (csv_header :: train);
  Utls.lines_to_file test_fn (csv_header :: test);
  (train_fn, test_fn)

let shuffle_then_cut seed p train_fn =
  match Utls.lines_of_file train_fn with
  | [] | [_] -> assert(false) (* no lines or header line only?! *)
  | (csv_header :: csv_payload) ->
    let rng = BatRandom.State.make [|seed|] in
    let rand_lines = L.shuffle ~state:rng csv_payload in
    let train, test = Cpm.Utls.train_test_split p rand_lines in
    train_test_dump csv_header train test

let shuffle_then_nfolds seed n train_fn =
  match Utls.lines_of_file train_fn with
  | [] | [_] -> assert(false) (* no lines or header line only?! *)
  | (csv_header :: csv_payload) ->
    let rng = BatRandom.State.make [|seed|] in
    let rand_lines = L.shuffle ~state:rng csv_payload in
    let train_tests = Cpm.Utls.cv_folds n rand_lines in
    L.rev_map (fun (x, y) -> train_test_dump csv_header x y) train_tests

(* what to do with the trained model *)
type mode = Load of string
          | Save of string
          | Discard

let trained_model_fn_from_mode = function
  | Discard -> failwith "Model.trained_model_fn_from_mode: discard"
  | Save _ -> failwith "Model.trained_model_fn_from_mode: save"
  | Load fn -> fn

let train verbose save_or_load config train_fn =
  match save_or_load with
  | Load trained_model_fn ->
    (Log.info "loading model from %s" trained_model_fn;
     trained_model_fn)
  | _ ->
    let model_fn = match save_or_load with
      | Load _ -> assert(false)
      | Save fn -> fn
      | Discard -> Fn.temp_file "orxgboost_model_" ".bin" in
    match Gblinear.train ~debug:verbose config train_fn with
    | Result.Error err -> failwith ("Model.train: " ^ err)
    | Result.Ok trained_model_fn ->
      begin
        Utls.run_command
          verbose (sprintf "mv %s %s" trained_model_fn model_fn);
        Log.debug "saving model to %s" model_fn;
        model_fn
      end

let test verbose model_fn test_fn =
  Gblinear.predict ~debug:verbose (Result.Ok model_fn) test_fn

let train_test_raw verbose save_or_load config train_fn test_fn =
  let model_fn = train verbose save_or_load config train_fn in
  let actual = extract_values verbose test_fn in
  let preds = test verbose model_fn test_fn in
  (model_fn, actual, preds)

let r2_plot no_plot actual preds =
  let test_R2 = Cpm.RegrStats.r2 actual preds in
  (if not no_plot then
     let title = sprintf "DNN model fit; R2=%.2f" test_R2 in
     Gnuplot.regr_plot title actual preds
  );
  Log.debug "R2_te: %.3f" test_R2;
  test_R2

let train_test verbose save_or_load no_plot config train_fn test_fn =
  let _model_fn, actual, preds =
    train_test_raw verbose save_or_load config train_fn test_fn in
  r2_plot no_plot actual preds

let decode_float_range (range_str: string): float list =
  L.map float_of_string
    (BatString.split_on_char ';' range_str)

let decode_int_range (range_str: string): int list =
  L.map int_of_string
    (BatString.split_on_char ';' range_str)

let main () =
  Log.(set_log_level INFO);
  Log.color_on ();
  let argc, args = CLI.init () in
  let train_portion_def = 0.8 in
  let show_help = CLI.get_set_bool ["-h";"--help"] args in
  if argc = 1 || show_help then
    begin
      eprintf "usage:\n\
               %s\n  \
               [--train <train.txt>]: training set\n  \
               [-p <float>]: train portion; default=%f\n  \
               [--seed <int>]: RNG seed\n  \
               [--test <test.txt>]: test set\n  \
               [--scan]: toggle scan of hyper params\n  \
               [--eta <float>]: learning rate in ]0.0:1.0]\n  \
               [--eta-scan [float;float;...]]: eta range\n  \
               [--lambda <float>]: L2 regularization in [0.0:100.0]\n  \
               [--lambda-scan \"float;float;...\": lambda range\n  \
               [--alpha <float>]: L1 regularization in [0.0:100.0]\n  \
               [--alpha-scan \"float;float;...\": alpha range\n  \
               [--rounds <int>]: number of training rounds >= 1\n  \
               [--rounds-scan \"int;int;...\": rounds range\n  \
               [-np <int>]: max CPU cores\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [-s <filename>]: save trained model to file\n  \
               [-l <filename>]: restore trained model from file\n  \
               [-o <filename>]: predictions output file\n  \
               [--no-plot]: don't call gnuplot\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this message\n"
        Sys.argv.(0) train_portion_def;
      exit 1
    end;
  let verbose = CLI.get_set_bool ["-v"] args in
  let must_scan = CLI.get_set_bool ["--scan"] args in
  let ncores = CLI.get_int_def ["-np"] args 1 in
  let seed = match CLI.get_int_opt ["--seed"] args with
    | Some s -> s (* reproducible *)
    | None -> (* random *)
      let () = Random.self_init () in
      Random.int 0x3FFFFFFF (* 0x3FFFFFFF = 2^30 - 1 *) in
  let no_plot = CLI.get_set_bool ["--no-plot"] args in
  let maybe_train_fn = CLI.get_string_opt ["--train"] args in
  let maybe_test_fn = CLI.get_string_opt ["--test"] args in
  let nrounds = CLI.get_int_def ["--rounds"] args 20 in
  Utls.enforce (nrounds >= 1) "nrounds < 1";
  (* eta = learning rate *)
  let eta = CLI.get_float_def ["--eta"] args 0.3 in
  Utls.enforce (0.0 < eta && eta <= 1.0) "eta not in ]0:1]";
  (* lambda = L2 regularization *)
  let lambda = CLI.get_float_def ["--lambda"] args 0.0 in
  Utls.enforce (lambda >= 0.0 && eta <= 100.0) "lambda not in [0.0:100]";
  (* alpha = L1 regularization *)
  let alpha = CLI.get_float_def ["--alpha"] args 0.0 in
  Utls.enforce (alpha >= 0.0 && alpha <= 100.0) "alpha not in [0.0:100]";
  (* all those default ranges are somewhat arbitrary *)
  let eta_range = match CLI.get_string_opt ["--eta-scan"] args with
    | Some s -> decode_float_range s
    | None -> [0.01; 0.02; 0.03; 0.05; 0.1; 0.2; 0.3; 0.5] in
  let lambda_range = match CLI.get_string_opt ["--lambda-scan"] args with
    | Some s -> decode_float_range s
    | None -> [0.01; 0.02; 0.03; 0.05; 0.1; 0.2; 0.3; 0.5; 1.0] in
  let alpha_range = match CLI.get_string_opt ["--alpha-scan"] args with
    | Some s -> decode_float_range s
    | None -> [0.01; 0.02; 0.03; 0.05; 0.1; 0.2; 0.3; 0.5; 1.0] in
  let rounds_range = match CLI.get_string_opt ["--rounds-scan"] args with
    | Some s -> decode_int_range s
    | None -> [10; 20; 30; 50; 100; 200; 300; 500; 1000; 2000; 3000; 5000] in
  let nfolds = CLI.get_int_def ["--NxCV"] args 1 in
  let train_portion = CLI.get_float_def ["-p"] args 0.8 in
  let scores_fn = match CLI.get_string_opt ["-o"] args with
    | None -> Fn.temp_file "orxgboost_preds_" ".txt"
    | Some fn -> fn in
  let save_or_load =
    match (CLI.get_string_opt ["-l"] args, CLI.get_string_opt ["-s"] args) with
    | (Some fn, None) -> Load fn
    | (None, Some fn) -> Save fn
    | (None, None) -> Discard
    | (Some _, Some _) -> failwith "Model: both -l and -s" in
  CLI.finalize ();
  let config = Gblinear.make_params eta lambda alpha nrounds in
  match maybe_train_fn, maybe_test_fn with
  | (None, None) -> failwith "provide --train and/or --test"
  | (None, Some test_fn) ->
    (* trained model production use *)
    let model_fn = trained_model_fn_from_mode save_or_load in
    let preds = test verbose model_fn test_fn in
    Utls.float_list_to_file scores_fn preds
  | (Some train_fn, Some test_fn) ->
    ignore(train_test verbose save_or_load no_plot config train_fn test_fn)
  | (Some train_fn', None) ->
    if nfolds > 1 then
      begin (* cross validation *)
        Log.info "shuffle -> %dxCV" nfolds;
        let train_test_fns = shuffle_then_nfolds seed nfolds train_fn' in
          let actual_preds =
            Parmap.parmap ncores (fun (train_fn, test_fn) ->
                train_test_raw
                  verbose save_or_load config train_fn test_fn
              ) train_test_fns in
          let actual = L.concat (L.map Utls.snd3 actual_preds) in
          let preds = L.concat (L.map Utls.trd3 actual_preds) in
          ignore(r2_plot no_plot actual preds)
      end
    else
      begin (* no cross validation *)
        (* train/test split *)
        Log.info "shuffle -> train/test split (p=%.2f)" train_portion;
        let train_fn, test_fn =
          shuffle_then_cut seed train_portion train_fn' in
        if not must_scan then
          let r2 =
            train_test verbose save_or_load no_plot config train_fn test_fn in
          Log.info "R2: %.3f" r2
        else
          let configs = ref [] in
          L.iter (fun e ->
              L.iter (fun l ->
                  L.iter (fun a ->
                      L.iter (fun n ->
                          configs := (e, l, a, n) :: !configs
                        ) rounds_range
                    ) alpha_range
                ) lambda_range
            ) eta_range;
          (* randomize them so that the parameter space exploration is not
             sequential/boring *)
          configs := L.shuffle !configs;
          Log.info "configs: %d" (L.length !configs);
          let (best_e, best_l, best_a, best_n, bets_r2) =
            Parany.Parmap.parfold ncores
              (fun (e, l, a, n) ->
                 let conf = Gblinear.make_params e l a n in
                 let r2 =
                   train_test verbose save_or_load no_plot conf train_fn test_fn in
                 (e, l, a, n, r2)
              )
              (fun (e, l, a, n, r2) (e', l', a', n', r2') ->
                 if r2' > r2 then
                   (Log.info "(e,l,a,n):r2 (%.2f, %.2f, %.2f, %d):%.3f"
                      e' l' a' n' r2';
                    (e', l', a', n', r2'))
                 else
                   (Log.warn "(e,l,a,n):r2 (%.2f, %.2f, %.2f, %d):%.3f"
                      e' l' a' n' r2';
                    (e, l, a, n, r2))
              ) (0.0, 0.0, 0.0, 0, 0.0) !configs in
          Log.info "BEST: (e,l,a,n):r2 (%.2f, %.2f, %.2f, %d):%.3f"
            best_e best_l best_a best_n bets_r2
      end

let () = main ()
