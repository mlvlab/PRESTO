export CUDA_VISIBLE_DEVICES=0

datasets=(
    antonyms
    auto_categorization
    auto_debugging
    cause_and_effect
    common_concept
    diff
    informal_to_formal
    letters_list
    negation
    object_counting
    odd_one_out
    orthography_starts_with
    rhymes
    second_word_letter
    sentence_similarity
    sum
    synonyms
    taxonomy_animal
    word_sorting
    word_unscrambling
 )

for i in "${datasets[@]}"; do
    for s in 3; do
        echo "Running task: $i, seed: $s, n_prompt_tokens: 5, intrinsic_dim: 50"
        python experiments/run_neural_bandits.py \
        --seed $s \
        --task $i \
        --target_model gpt \
        --n_prompt_tokens 5 \
        --nu 1 \
        --lamdba 0.1 \
        --n_init 40 \
        --n_domain 10000 \
        --total_iter 165 \
        --local_training_iter 1000 \
        --n_eval 1000 \
        --intrinsic_dim 50 \
        --gpt gpt-4.1 \
        --name iter165
    done
done