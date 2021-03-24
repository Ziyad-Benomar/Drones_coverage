class ActionMapping:
    mapping = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

    @staticmethod
    def to_coords(current, action):
        action_mapping = ActionMapping.mapping[action]
        return (current[0] + action_mapping[0], current[1] + action_mapping[1])


def get_cell_state_v2(info_map, bump_map, k, l, drone_coords, action_size):
    action_coords = [ActionMapping.to_coords(drone_coords, a) for a in range(action_size)]
    info_coord = [None] * 9
    info_coord[0]=0
    for s in range(9):
        i, j = action_coords[s]
        if (i >= 0 and j >= 0 and i < k and j < l):
            info_coord[s] = info_map[i, j]

    right_cell = action_coords[1]
    left_cell = action_coords[5]
    top_cell = action_coords[3]
    bottom_cell = action_coords[7]

    i, j = right_cell
    if (i >= 0 and j >= 0 and i < k and j < l and bump_map[j, i] < 0):
        info_coord[2] = None
        info_coord[8] = None
    i, j = left_cell
    if (i >= 0 and j >= 0 and i < k and j < l and bump_map[j, i] < 0):
        info_coord[4] = None
        info_coord[6] = None
    i, j = top_cell
    if (i >= 0 and j >= 0 and i < k and j < l and bump_map[j, i] < 0):
        info_coord[2] = None
        info_coord[4] = None
    i, j = bottom_cell
    if (i >= 0 and j >= 0 and i < k and j < l and bump_map[j, i] < 0):
        info_coord[6] = None
        info_coord[8] = None

    for a in range(action_size):
        i, j = ActionMapping.to_coords(drone_coords, a)
        if i < 0 or j < 0 or i >= k or j >= l or bump_map[j, i] < 0:
            info_coord[a] = None

    return info_coord


def tuple_to_coord(tuple):
    return tuple[0], tuple[1]
