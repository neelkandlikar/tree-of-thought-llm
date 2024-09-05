import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    
    valid_proposals = [
        proposal for proposal in proposals 
        if not any(phrase in proposal.lower() for phrase in [
            "no operations can be performed",
            "no possible next steps",
            "single number",
        ])
    ]
    
    if not valid_proposals:
        return []
    
    # Otherwise, return the valid proposals
    return [y + _ + '\n' for _ in valid_proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve_bfs(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}



def dfs_helper(args, task, idx, y, t, paths, infos):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    T = args.T 
    
    step_info = {'step': t, 'x': x, 'y': y}
    
    if t >= T:
        # Evaluate final reasoning
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, [y], args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, [y], args.n_evaluate_sample)
        
        current_value = values[0] if values else 0
        
        step_info['values'] = values
        step_info['selected'] = []
        
        if current_value >= args.thresh:
            paths.append((y, current_value))
            step_info['selected'] = [y]
        
        infos.append(step_info)
        return
    
    # Generation
    if args.method_generate == 'sample':
        new_ys = get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample)
    elif args.method_generate == 'propose':
        new_ys = get_proposals(task, x, y)
    print(new_ys)
    
    step_info['new_ys'] = new_ys
    
    # Evaluation
    if args.method_evaluate == 'vote':
        values = get_votes(task, x, new_ys, args.n_evaluate_sample)
    elif args.method_evaluate == 'value':
        values = get_values(task, x, new_ys, args.n_evaluate_sample)
    print(values)
    
    step_info['values'] = values

    candidates = sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
    candidates = [(new_y, value) for new_y, value in candidates if value >= args.thresh]

    selected = [new_y for new_y, _ in candidates[:args.n_select_sample]]
    step_info['selected'] = selected
    
    infos.append(step_info)

    for new_y, value in candidates[:args.n_select_sample]:
        dfs_helper(args, task, idx, new_y, t+1, paths, infos)

def solve_dfs(args, task, idx):
    paths = []
    infos = []
    dfs_helper(args, task, idx, '', 0, paths, infos)
    return paths, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}