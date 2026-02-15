import random
import torch
import sys
import os
import json
import numpy as np
cwd = os.getcwd()
sys.path.append(cwd)
from automatic_prompt_engineer import data
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from automatic_prompt_engineer import evaluate, config, template, data
import re

from torch.quasirandom import SobolEngine
from tqdm import tqdm
import argparse
from experiments.evaluation.instruction_induction.utility import set_all_seed, TASKS
from itertools import accumulate

import gpytorch
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.acquisition.analytic import UpperConfidenceBound, LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from experiments.group_init import greedy_group_selection
from experiments.score_equality_utils import build_constraint_matrix
from experiments.constrained_gp import ConstrainedGP
from LlamaForMLPRegression import NeuralTSDiag
import time

SMOKE_TEST = os.environ.get("SMOKE_TEST")
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

model_name = "llama"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_instruction_by_embedding(instruct_to_embedding, target_embedding):
    for instruction, embeddings in instruct_to_embedding.items():
        for emb in embeddings:
            if torch.equal(emb, target_embedding):
                return instruction
    breakpoint()
    return None 

class LMForwardAPI:
    def __init__(self, model_name='llama', eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, n_prompt_tokens=None, few_shot_data=None, 
                 HF_cache_dir=None, random_proj=None, intrinsic_dim=None, target_model=None):
        p = torch.ones(10)

        kwargs={'torch_dtype': torch.float16}
        if model_name in ["llama"]:
            self.model = LlamaForCausalLM.from_pretrained(
                                HF_cache_dir, low_cpu_mem_usage=True, device_map="auto", **kwargs
                            )
            self.model.eval()
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p       = None

            self.tokenizer = AutoTokenizer.from_pretrained(
                                HF_cache_dir,
                                model_max_length=512,
                                use_fast=False,
                            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise NotImplementedError

        init_token = init_prompt[0] + init_qa[0]
        self.init_token = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": init_token}
        ]

        if model_name in ["vicuna", "llama", "mistral"]:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]
            
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)
        for p in self.linear.parameters():   
            torch.nn.init.uniform_(p, -1, 1)

        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")
        
        if target_model in ['gpt']:
            self.api_model = exec_evaluator(target_model, self.conf)

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.prompts_list = []
        self.target_model = target_model

    @torch.no_grad()
    def get_representation_batch(self, prompt_embeddings): # (batch_size, token_num, 4096)
        batch_size = prompt_embeddings.shape[0]
        prompt_embeddings = prompt_embeddings.reshape(batch_size, self.n_prompt_tokens, -1)

        token_id_lists = self.tokenizer.apply_chat_template(self.init_token)
        batch_token_id_lists = [token_id_lists.copy() for _ in range(batch_size)]

        enc = self.tokenizer.pad(
            {"input_ids": batch_token_id_lists},
            padding=True,        
            return_tensors="pt", 
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        input_embed = self.embedding[input_ids]
        input_embed = torch.cat((prompt_embeddings, input_embed), dim=1)

        soft_mask = torch.ones(batch_size, prompt_embeddings.size(1), dtype=attention_mask.dtype)
        attention_mask = torch.cat([soft_mask, attention_mask], dim=1).to(device=input_embed.device)
        outputs = self.model.generate(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=64,
            do_sample=False
        )

        output_embeddings = outputs.hidden_states[0][-1][:, -1, :]
        output_ids = outputs.sequences
        instructions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        instructions = [s.removeprefix("assistant\n\n") for s in instructions]
        instructions = [s.removeprefix("assistant\n") for s in instructions]
        instructions = [s.removeprefix("assistant") for s in instructions]

        return output_embeddings, instructions
    
    def eval(self, instruction):
        self.num_call += 1
        print('Instruction: {}'.format(instruction))
        
        if instruction in self.prompts_set.keys():
            dev_perf = self.prompts_set[instruction]
        else:
            if self.target_model in ['gpt']: 
                dev_perf = self.api_model.evaluate(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']).sorted()[1][0]
                self.prompts_set[instruction] = dev_perf
            else:
                raise NotImplementedError
        self.prompts_list.append((len(self.prompts_list), instruction, dev_perf))

        if dev_perf >= self.best_last_perf:
            self.count += 1
        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_instruction = instruction

        print('{}th score: {} Best score: {}'.format(self.num_call, round(float(dev_perf), 4), round(float(self.best_dev_perf), 4)))
        print('--------------------------------------------------------------------------------')

        return dev_perf.item()

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set

    def return_prompts_list(self):
        return self.prompts_list
    
def run(task, n_prompt_tokens, HF_cache_dir, nu, lamdba, n_init, n_domain, total_iter, local_training_iter, random_proj, intrinsic_dim, n_eval, gpt, init_scale, pooling, target_model, n_batch, seed):
    assert task in TASKS, 'Task not found!'

    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]

    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    init_prompt = ['\n']
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to"

    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        }
    }

    subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]

    model_forward_api = LMForwardAPI(model_name=model_name, eval_data=eval_data, init_prompt=init_prompt, 
                                    init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data,
                                    n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir, random_proj=random_proj,intrinsic_dim=intrinsic_dim, target_model=target_model)

    all_X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(n_domain)
    with torch.no_grad():
        all_X = model_forward_api.linear(all_X)

    all_X = all_X.to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)
    instruct_to_embedding = dict() # {'instruction1': torch tensor (n_1, d), 'instruction2': torch tensor (n_2, d), ...}
    batch_size = 256

    while True:
        try:
            tilde_Z = torch.zeros([0]).to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)
            instruct_to_embedding = {}
            
            for i in tqdm(range(0, len(all_X), batch_size), desc="Get instruction"):
                batch_X = all_X[i:i + batch_size]
                output_embeddings, instructions = model_forward_api.get_representation_batch(batch_X)
                for instr, emb in zip(instructions, output_embeddings):
                    if instr not in instruct_to_embedding:
                        instruct_to_embedding[instr] = torch.zeros([0]).to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)
                    instruct_to_embedding[instr] = torch.cat((instruct_to_embedding[instr], emb[None,:]), dim=0)
                    tilde_Z = torch.cat((tilde_Z, emb[None,:]), dim=0)
            break 
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch size {batch_size}, retrying with smaller batch")
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                if batch_size < 1:
                    raise ValueError("batch size too small")
            else:
                raise e
    torch.cuda.empty_cache()
    
    filtered_items = [(k, v) for k, v in instruct_to_embedding.items() if v.size(0) >= 5] # Efficiency
    if len(filtered_items) <= n_init:
        observed_instructions = sorted(instruct_to_embedding, key=lambda k: instruct_to_embedding[k].numel(), reverse=True)[:40]
    else:
        keys, groups = zip(*filtered_items)
        observed_instructions = greedy_group_selection(list(groups), list(keys), tilde_Z, n_init) # MMD
    assert len(observed_instructions) == n_init, f"Selected {len(observed_instructions)} instructions, but expected {n_init}."
        
    observed_scores = [model_forward_api.eval(instruction) for instruction in observed_instructions]
    X_train = torch.zeros([0]).to(tkwargs['device'])
    y_train = []
    for i, instruction in enumerate(observed_instructions): # Score assignment
        X_train = torch.cat((X_train, instruct_to_embedding[instruction]), dim=0) 
        y_train += [observed_scores[i]] * (len(instruct_to_embedding[instruction]))
    X_train = X_train.to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)
    y_train = torch.tensor(y_train).unsqueeze(-1).to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)
    print(f"Number of initial data: {len(X_train)}")
    for instruction in observed_instructions:
        instruct_to_embedding.pop(instruction)

    emb_list = [v for v in instruct_to_embedding.values()]
    X_unseen = torch.cat(emb_list, dim=0).to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)

    max_iter = total_iter - n_init
    l = NeuralTSDiag(input_dim=model_forward_api.hidden_size, lamdba=lamdba, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=True)
    instruct_to_embedding_larger_two = {k: v.to(**tkwargs) for k, v in instruct_to_embedding.items() if len(v) >= 2}
    l.train(None, None, instruct_to_embedding_larger_two, local_training_iter)
    all_embeddings = torch.cat(list(instruct_to_embedding_larger_two.values()), dim=0)
    l.update_U(all_embeddings)
    
    best_r = y_train.max().item()
    for t in range(max_iter):
        selected_idx = np.random.choice(len(X_unseen), min(len(X_unseen), n_eval), replace=False)
        arm_select = l.select(X_unseen[selected_idx])
        selected_embedding = X_unseen[selected_idx][arm_select]
        next_instruction = find_instruction_by_embedding(instruct_to_embedding, selected_embedding)
        
        r = model_forward_api.eval(next_instruction)
        observed_scores.append(r)
        best_r = max(r, best_r)

        next_X = instruct_to_embedding[next_instruction]
        if len(next_X) != 1:
            next_X = next_X
            mask = ~(next_X == selected_embedding).all(dim=1)
            masked_next_X = next_X[mask]
            l.update_U(masked_next_X)
        next_y = [r] * (len(next_X))
        next_y = torch.tensor(next_y).unsqueeze(-1).to(**tkwargs)
        instruct_to_embedding.pop(next_instruction)
        
        emb_list = [v for v in instruct_to_embedding.values()]
        X_unseen = torch.cat(emb_list, dim=0).to(device=model_forward_api.embedding.device, dtype=model_forward_api.embedding.dtype)

        instruct_to_embedding_larger_two = {k: v.to(**tkwargs) for k, v in instruct_to_embedding.items() if len(v) >= 2}
        l.train(next_X, next_y, instruct_to_embedding_larger_two, local_training_iter)
        if best_r == 1.0:
            print('Find the best instruction!')
            break

    best_values = list(accumulate(observed_scores, max))
    prompts = model_forward_api.return_best_prompt()
    prompts_list = model_forward_api.return_prompts_list()

    print('Evaluating on test data...')
    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(100, len(test_data[0])),
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                   'model': gpt
                }
            }
        }
    }
    num_scores = len(X_train)
    test_score = model_forward_api.api_model.test(prompts, model_forward_api.eval_template, test_data, test_conf).sorted()[1][0]

    return test_score, prompts, prompts_list, best_values, num_scores

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Set the parameter nu."    
    )
    parser.add_argument(
        "--lamdba",
        type=float,
        default=0.1,
        help="Set the lamdba parameter."    
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=40,
        help="Set the number of initialization points."    
    )
    parser.add_argument(
        "--n_domain",
        type=int,
        default=10000,
        help="Set the number of domain."    
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=165,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--local_training_iter",
        type=int,
        default=30,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default='uniform',
        help="Set the projection method."    
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=100,
        help="Set the number of intrinsic dim."    
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="Set the number of domains to be evaluated at each ucb iteration."
    )
    parser.add_argument(
        "--n_batch",
        type=int,
        default=25,
    )    
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Set the name of the experiments."    
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="gpt-3.5-turbo",
        help="Which version of gpt to use."    
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=1,
        help="Which scale to use."    
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        help="Which pooling method to use."    
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="gpt"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(args.seed))
    test_score, prompts, prompts_list, best_values, num_scores = run(
        task=args.task,
        n_prompt_tokens=args.n_prompt_tokens,
        HF_cache_dir=args.HF_cache_dir,
        nu=args.nu,
        lamdba=args.lamdba,
        n_init=args.n_init,
        n_domain=args.n_domain,
        total_iter=args.total_iter,
        local_training_iter = args.local_training_iter,
        random_proj=args.random_proj,
        intrinsic_dim=args.intrinsic_dim,
        n_eval=args.n_eval,
        gpt=args.gpt,
        init_scale=args.init_scale,
        pooling=args.pooling,
        target_model=args.target_model,
        n_batch=args.n_batch,
        seed=args.seed
    )
    
    args_dict = vars(args)
    args_dict['test_score'] = test_score
    args_dict['best_prompt'] = prompts
    args_dict['prompts_list'] = prompts_list
    args_dict['best_values'] = best_values
    args_dict['num_scores'] = num_scores

    save_dir = f"./Results/{args.task}"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, args.target_model + str(args.seed) + ".json")

    with open(path, 'x') as fp:
        json.dump(args_dict, fp, indent=4)
    
    print(f'Test score on ChatGPT: {test_score}')
    print("Finished!!!")