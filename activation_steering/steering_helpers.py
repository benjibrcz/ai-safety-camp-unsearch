import torch
from matplotlib.gridspec import GridSpec
from unsearch_research import *
from unsearch_research.stable.analysis import intervene_upon_generation

def find_vector(tokenizer, model, DEVICE, data_set, layer, shape):

    cache_n = torch.zeros(shape)
    cache_n = cache_n.to(DEVICE)

    for i in range(len(data_set)):
        # Prepare example and hook
        example = data_set[i].as_tokens(tokenizer)
        prompt, soln = example[: example.index("<PATH_START>") + 1], example[example.index("<PATH_START>") + 1 :]

        name = f"blocks.{layer}.hook_resid_pre"
        cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
        with model.hooks(fwd_hooks=caching_hooks):
            _ = model(" ".join(prompt))
        cache_n+= cache[name]/len(data_set)

    return cache_n

def find_vector_at_step_s(s, tokenizer, model, DEVICE, data_set, layer, shape):

    cache_n = torch.zeros(shape)
    cache_n = cache_n.to(DEVICE)

    for i in range(len(data_set)):
        # Prepare example and hook
        example = data_set[i].as_tokens(tokenizer)
        prompt, soln = example[: example.index("<PATH_START>") + s], example[example.index("<PATH_START>") + s :]

        name = f"blocks.{layer}.hook_resid_pre"
        cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
        with model.hooks(fwd_hooks=caching_hooks):
            _ = model(" ".join(prompt))
        cache_n+= cache[name]/len(data_set)

    return cache_n

def find_vector_all(tokenizer, model, DEVICE, data_set, num_layers, shape):

    vecs = []
    for layer in range(num_layers):
        cache_n = torch.zeros(shape)
        cache_n = cache_n.to(DEVICE)

        for i in range(len(data_set)):
            # Prepare example and hook
            example = data_set[i].as_tokens(tokenizer)
            prompt, soln = example[: example.index("<PATH_START>") + 1], example[example.index("<PATH_START>") + 1 :]

            name = f"blocks.{layer}.hook_resid_pre"
            cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
            with model.hooks(fwd_hooks=caching_hooks):
                _ = model(" ".join(prompt))
            cache_n+= cache[name]/len(data_set)

        vecs.append(cache_n)

    return vecs

def find_vector_all_at_step_s(s, tokenizer, model, DEVICE, data_set, num_layers, shape):

    vecs = []
    for layer in range(num_layers):
        cache_n = torch.zeros(shape)
        cache_n = cache_n.to(DEVICE)

        for i in range(len(data_set)):
            # Prepare example and hook
            example = data_set[i].as_tokens(tokenizer)
            prompt, soln = example[: example.index("<PATH_START>") + s], example[example.index("<PATH_START>") + s :]

            name = f"blocks.{layer}.hook_resid_pre"
            cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
            with model.hooks(fwd_hooks=caching_hooks):
                _ = model(" ".join(prompt))
            cache_n+= cache[name]/len(data_set)

        vecs.append(cache_n)

    return vecs

def find_vector_all_layers_multiple_mazes(tokenizer, model, DEVICE, data_set, num_layers, shape):

    vecs = []
    for layer in range(num_layers):
        print(f'Starting layer {layer}')
        cache_n = torch.zeros(shape)
        cache_n = cache_n.to(DEVICE)
        for d in data_set:
            for i in range(len(d)):
                # Prepare example and hook
                example = d[i].as_tokens(tokenizer)
                prompt, soln = example[: example.index("<PATH_START>") + 1], example[example.index("<PATH_START>") + 1 :]

                name = f"blocks.{layer}.hook_resid_pre"
                cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
                with model.hooks(fwd_hooks=caching_hooks):
                   _ = model(" ".join(prompt))
                cache_n+= cache[name]/len(d)

        vecs.append(cache_n/len(data_set))

    return vecs

def find_vector_all_layers_multiple_mazes_at_step_s(s, tokenizer, model, DEVICE, data_set, num_layers, shape):

    vecs = []
    for layer in range(num_layers):
        print(f'Starting layer {layer}')
        cache_n = torch.zeros(shape)
        cache_n = cache_n.to(DEVICE)
        for d in data_set:
            for i in range(len(d)):
                # Prepare example and hook
                example = d[i].as_tokens(tokenizer)
                prompt, soln = example[: example.index("<PATH_START>") + s], example[example.index("<PATH_START>") + s :]

                name = f"blocks.{layer}.hook_resid_pre"
                cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
                with model.hooks(fwd_hooks=caching_hooks):
                   _ = model(" ".join(prompt))
                cache_n+= cache[name]/len(d)

        vecs.append(cache_n/len(data_set))

    return vecs

def create_activation_vector(tokenizer, model, DEVICE, pos_data_set, neg_data_set, layer, shape):

    pos_vec = find_vector(tokenizer, model, DEVICE, pos_data_set, layer, shape)
    neg_vec = find_vector(tokenizer, model, DEVICE, neg_data_set, layer, shape)

    return pos_vec-neg_vec

def create_activation_vector_multiple(tokenizer, model, DEVICE, pos_data_set, neg_data_set, layer, shape, mean_vec = None):
    act = torch.zeros(shape).to(DEVICE)
    i=0
    for p,n in zip(pos_data_set, neg_data_set):
        #print(f"Working on i = {i}")
        act += create_activation_vector(tokenizer, model, DEVICE, p, n, layer, shape)
        i=i+1

    act = act/len(pos_data_set)

    if mean_vec != None:
        act = act - mean_vec

    return act

def create_activation_vector_all_layers_multiple_mazes(tokenizer, model, DEVICE, pos_data_set, neg_data_set, num_layers, shape, mean_vecs = None):

    pos_vecs = find_vector_all_layers_multiple_mazes(tokenizer, model, DEVICE, pos_data_set, num_layers, shape)
    neg_vecs = find_vector_all_layers_multiple_mazes(tokenizer, model, DEVICE, neg_data_set, num_layers, shape)

    act_vects = [(p-n) for p,n in zip(pos_vecs, neg_vecs)]

    if mean_vecs != None:
        act_vects = [(a-m) for a,m in zip(act_vects, mean_vecs)]

    return act_vects

def create_activation_vector_all_layers_multiple_mazes_at_step_s(s, tokenizer, model, DEVICE, pos_data_set, neg_data_set, num_layers, shape, mean_vecs = None):

    pos_vecs = find_vector_all_layers_multiple_mazes_at_step_s(s, tokenizer, model, DEVICE, pos_data_set, num_layers, shape)
    neg_vecs = find_vector_all_layers_multiple_mazes_at_step_s(s, tokenizer, model, DEVICE, neg_data_set, num_layers, shape)

    act_vects = [(p-n) for p,n in zip(pos_vecs, neg_vecs)]

    if mean_vecs != None:
        act_vects = [(a-m) for a,m in zip(act_vects, mean_vecs)]

    return act_vects

def create_mean_vector_multiple(tokenizer, model, DEVICE, data_set, layer, shape):

    vec = torch.zeros(shape).to(DEVICE)

    for d in data_set:
        vec += find_vector(tokenizer, model, DEVICE, d, layer, shape)

    vec = vec/len(data_set)

    return vec

def create_mean_vector_multiple_all(tokenizer, model, DEVICE, data_set, num_layers, shape):

    vec = torch.zeros(shape).to(DEVICE)

    for d in data_set:
        vec += find_vector_all(tokenizer, model, DEVICE, d, num_layers, shape)

    vec = vec/len(data_set)

    return vec

def create_mean_vector_all_layers_multiple_mazes(tokenizer, model, DEVICE, data_set, num_layers, shape):

    vecs = find_vector_all_layers_multiple_mazes(tokenizer, model, DEVICE, data_set, num_layers, shape)

    return vecs

def create_mean_vector_all_layers_multiple_mazes_at_step_s(s, tokenizer, model, DEVICE, data_set, num_layers, shape):

    vecs = find_vector_all_layers_multiple_mazes_at_step_s(s, tokenizer, model, DEVICE, data_set, num_layers, shape)

    return vecs

def patching_hook_all_tokens(layer_to_patch, coeff, act):
    # Model in is list of str tokens (could batch tensor and use gather)
    def layer_output_patch_hook(inp, hook):
        # inp has shape [batch_size, seq_len, hidden_dim]

        if inp.shape[1] == 1:
            return

        ppos, apos = inp.shape[1], act.shape[1]

        if apos <= ppos:
            inp[:, :apos, :] += coeff * act
        else:
            inp += coeff * act[:,apos-ppos:]

        return inp

    return [(f"blocks.{layer_to_patch}.hook_resid_post", layer_output_patch_hook)]


def patching_hook_one_token(model_in, token_to_patch, layer_to_patch, coeff, act):
    # Model in is list of str tokens (could batch tensor and use gather)
    def layer_output_patch_hook(inp, hook):
        # inp has shape [batch_size, seq_len, hidden_dim]

        matched_pos = model_in.index(token_to_patch)

        # patch inp at matched_pos
        inp[:,matched_pos] += coeff * act[:,matched_pos+1] #5 * torch.randn_like(inp[:, 0, :])#
        #print(inp)
        return inp

    return [(f"blocks.{layer_to_patch}.hook_resid_post", layer_output_patch_hook)]

def apply_act_vecs(tokenizer, model, DEVICE, act_vec, layer, coeff, example):

    prompt, soln = example[: example.index("<PATH_START>") + 1], example[example.index("<PATH_START>") + 1 :]

    turn_right_patch_hook = patching_hook_one_token(prompt, example[example.index("<PATH_START>") + 2], layer, coeff, act_vec)

    prediction_original = model.generate(
        "".join(prompt),
        prepend_bos=False,
        max_new_tokens=40,
        do_sample=False,
        verbose=False,
    )

    prediction_perturbed = intervene_upon_generation(
        model,
        "".join(prompt),
        turn_right_patch_hook,
        do_sample=False,
        max_new_tokens=40,
        device=DEVICE,
        verbose=False,
    )
    prediction_perturbed = model.to_string(prediction_perturbed)[0]

    return prediction_original, prediction_perturbed

def apply_act_vecs_at_step_s(s, tokenizer, model, DEVICE, act_vec, layer, coeff, example):

    prompt, soln = example[: example.index("<PATH_START>") + s], example[example.index("<PATH_START>") + s :]

    turn_to_dir_patch_hook = patching_hook_one_token(prompt, example[example.index("<PATH_START>") + s + 1], layer, coeff, act_vec)

    prediction_original = model.generate(
        "".join(prompt),
        prepend_bos=False,
        max_new_tokens=40,
        do_sample=False,
        verbose=False,
    )

    prediction_perturbed = intervene_upon_generation(
        model,
        "".join(prompt),
        turn_to_dir_patch_hook,
        do_sample=False,
        max_new_tokens=40,
        device=DEVICE,
        verbose=False,
    )
    prediction_perturbed = model.to_string(prediction_perturbed)[0]

    return prediction_original, prediction_perturbed

def plot_intervened_mazes(tokenizer, prediction_original, prediction_perturbed):
    gs = GridSpec(1, 2)
    plot_maze(prediction_original, tokenizer=tokenizer, ax=gs[0, 0], ax_labels={'maze':{'title': 'Original'}})
    plot_maze(prediction_perturbed, tokenizer=tokenizer, ax=gs[0, 1], ax_labels={'maze':{'title': 'Perturbed'}});