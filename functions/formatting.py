def get_subgroup_str(attrs):
    subgroup = []
    if attrs:
        for col in attrs:
            subgroup.append(col)
    else:
        subgroup.append("Overall")

    return ', '.join(subgroup)

def get_indices(subgroups, X):
    subgroup_inds = []
    subgroup_names = []
    for k, v in subgroups.items():
        for c in v:
            curr_inds = []
            for g in c:
                curr_inds.append((X.columns.get_loc(g[0]), g[1]))

            subgroup_inds.append(curr_inds)
            subgroup_names.append(c)

    return subgroup_inds, subgroup_names