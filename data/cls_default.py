# metadata for tasks, labels, and races
tasks = ['NSCLC_subtyping', 'BRCA_subtyping', 'Ebrains_IDH1Mutation']
label_dicts = {
    'NSCLC_subtyping': {'LUAD': 0, 'LUSC': 1},
    'BRCA_subtyping': {'IDC': 0, 'ILC': 1},
    'Ebrains_IDH1Mutation' : {"idhwt": 0, "idhmut": 1},
    'race_map': {"W": 0, "B": 1, "A": 2, "N": 3, "H":4},
}