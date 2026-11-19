#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
usage() { printf 'Usage: bash launchers/sweep.sh --gpus 0,1,2,3,4 --max-concurrency N [--execute]\n'; }
gpus=""; max_concurrency=1; execute=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --max-concurrency) max_concurrency="$2"; shift 2 ;;
    --execute) execute=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$gpus" ]] || die "sweep requires --gpus"
[[ "$max_concurrency" =~ ^[1-5]$ ]] || die "--max-concurrency must be at most 5"
IFS=',' read -ra gpu_ids <<<"$gpus"
(( max_concurrency <= ${#gpu_ids[@]} )) || die "max concurrency exceeds selected GPU count"
for gpu in "${gpu_ids[@]}"; do validate_gpu_permission "$gpu"; done
methods=(skew_trained learned_orthogonal learned_conditional_rotation)
prompts=(golden_retriever domestic_cat horse parrot butterfly red_fox wooden_chair sports_car)
seeds=(17 18 19)
declare -A prompt_text=(
  [golden_retriever]='A photograph of a golden retriever.'
  [domestic_cat]='A photograph of a domestic cat.'
  [horse]='A photograph of a horse.'
  [parrot]='A photograph of a parrot.'
  [butterfly]='A photograph of a butterfly.'
  [red_fox]='A photograph of a red fox.'
  [wooden_chair]='A photograph of a wooden chair.'
  [sports_car]='A photograph of a sports car.'
)
printf '%s\n' "$([[ "$execute" == "true" ]] && printf 'EXECUTE' || printf 'DRY RUN')"
index=0; active=0; failures=0
for prompt in "${prompts[@]}"; do
  for seed in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
      gpu="${gpu_ids[$((index % ${#gpu_ids[@]}))]}"
      run_id="sweep-${prompt}-s${seed}-${method}"
      output="$PROJECT_ROOT/outputs/sweep/$run_id"
      printf 'run_id=%s gpu=%s output=%s method=%s prompt=%s seed=%s\n' "$run_id" "$gpu" "$output" "$method" "$prompt" "$seed"
      if [[ "$execute" == "true" ]]; then
        bash "$PROJECT_ROOT/launchers/train.sh" --gpu "$gpu" --method "$method" --run-id "$run_id" --output-root "$PROJECT_ROOT/outputs/sweep" --override "experiment.prompt_id=$prompt" --override "experiment.prompt=${prompt_text[$prompt]}" --override "experiment.training_seed=$seed" &
        ((active+=1))
        if (( active >= max_concurrency )); then
          wait -n || ((failures+=1))
          ((active-=1))
        fi
      fi
      ((index+=1))
    done
  done
done
if [[ "$execute" == "true" ]]; then
  while (( active > 0 )); do wait -n || ((failures+=1)); ((active-=1)); done
  printf 'child_failures=%s\n' "$failures"
  (( failures == 0 ))
else
  printf 'projected_cost=unavailable_until_profile\n'
fi
