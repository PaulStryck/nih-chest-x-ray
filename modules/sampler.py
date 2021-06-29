def weights(dataset):
    _img, _target = dataset[0]
    combined_targets  = [0]*len(_target)
    weights = []
    for _, t in dataset:
        combined_targets = [sum(x) for x in zip(combined_targets, t)]

    for _, t in dataset:
        prevalences = [x[0] for x in zip(combined_targets, t) if x[1] != 0]
        weights.append(sum(prevalences))

    return [(1/x)*1000 for x in weights]
