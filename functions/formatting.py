def get_subgroup_str(attrs):
    subgroup = []
    if attrs:
        for col, val in attrs:
            if col == 'country_cd_US':
                subgroup.append("US" if val == 1 else "International")
            elif col == 'is_female':
                subgroup.append("Female" if val == 1 else "Male/Other")
            elif col == 'bachelor_obtained':
                subgroup.append("Bachelor or higher" if val == 1 else "No Bachelor")
            else:
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