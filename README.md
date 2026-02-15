# PRESTO
[NeurIPS 25] Official Implementation (Pytorch) of "[PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](https://arxiv.org/pdf/2510.25808)"

Our code are based on the code from [APE](https://github.com/keirp/automatic_prompt_engineer), [InstructZero](https://github.com/Lichang-Chen/InstructZero) and [INSTINCT](https://github.com/xqlin98/INSTINCT).

### Prepare the data
You can download the data for intrinsic induction from the github repo of [InstructZero](https://github.com/Lichang-Chen/InstructZero). You can download the dataset for GSM8K, AQUARAT, and SVAMP from the repo for [APE](https://github.com/keirp/automatic_prompt_engineer).

# Run our code
To run our code, you need to install the environment using conda:
`conda env create -f environment.yml`

We provide bash scripts for running our experiments for instruction induction at `Induction/experiments/run_neural_bandits.sh`. To run it properly, you need to run the following in the terminal:
```
cd Induction
bash experiments/run_neural_bandits.sh
```
