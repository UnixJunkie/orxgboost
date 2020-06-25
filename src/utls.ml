
open Printf

module L = BatList
module Log = Dolog.Log

let with_in_file fn f =
  let input = open_in_bin fn in
  let res = f input in
  close_in input;
  res

let with_out_file fn f =
  let output = open_out_bin fn in
  let res = f output in
  close_out output;
  res

let lines_of_file fn =
  with_in_file fn (fun input ->
      let res, exn = L.unfold_exc (fun () -> input_line input) in
      if exn <> End_of_file then
        raise exn
      else res
    )

let lines_to_file fn lines =
  with_out_file fn (fun out ->
      L.iter (fprintf out "%s\n") lines
    )

(* call f on lines of file *)
let iter_on_lines_of_file fn f =
  let input = open_in_bin fn in
  try
    while true do
      f (input_line input)
    done
  with End_of_file -> close_in input

let count_lines (fn: string): int =
  let count = ref 0 in
  iter_on_lines_of_file fn (fun _line ->
      incr count
    );
  !count

let append_file_to_buffer buff fn =
  with_in_file fn (fun input ->
      let len = in_channel_length input in
      Buffer.add_channel buff input len
    )

let ignore_fst _fst snd =
  snd

let fold_on_lines_of_file fn f acc =
  with_in_file fn (fun input ->
      let acc' = ref acc in
      try
        while true do
          acc' := f !acc' (input_line input)
        done;
        assert(false)
      with End_of_file -> !acc'
    )

let float_list_of_file fn =
  let res =
    fold_on_lines_of_file fn (fun acc line ->
        let pred =
          try Scanf.sscanf line "%f" (fun x -> x)
          with Scanf.Scan_failure msg ->
            (* percolate a NaN rather than crashing *)
            (Log.error "%s: %s" msg line;
             nan) in
        pred :: acc
      ) [] in
  L.rev res

let float_list_to_file fn l =
  with_out_file fn (fun out ->
      L.iter (fprintf out "%f\n") l
    )

type filename = string

(* capture everything in case of error *)
let collect_script_and_log
    (debug: bool)
    (r_script_fn: filename) (r_log_fn: filename) (model_fn: filename)
  : Result.t =
  let buff = Buffer.create 4096 in
  bprintf buff "--- %s ---\n" r_script_fn;
  append_file_to_buffer buff r_script_fn;
  bprintf buff "--- %s ---\n" r_log_fn;
  append_file_to_buffer buff r_log_fn;
  let err_msg = Buffer.contents buff in
  if not debug then L.iter Sys.remove [r_script_fn; r_log_fn; model_fn];
  Error err_msg

let read_predictions (debug: bool) (maybe_predictions_fn: Result.t): float list =
  match maybe_predictions_fn with
  | Error err -> failwith err (* should have been handled by user before *)
  | Ok predictions_fn ->
    if debug then Log.debug "%s" predictions_fn;
    let res = float_list_of_file predictions_fn in
    if not debug then Sys.remove predictions_fn;
    res

let run_command verbose cmd =
  if verbose then Log.info "cmd: %s" cmd;
  ignore(Sys.command cmd)

let fst3 (a, _, _) = a
let snd3 (_, b, _) = b
let trd3 (_, _, c) = c

(* abort if condition is not met *)
let enforce (condition: bool) (err_msg: string): unit =
  if not condition then
    failwith err_msg
