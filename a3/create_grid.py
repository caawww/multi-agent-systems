import yaml


def radius_from_level(level, min_level=1, max_level=4,
                      min_radius=0.18, max_radius=0.45):
    level = max(min_level, min(max_level, level))
    if max_level == min_level:
        return max_radius

    t = (level - min_level) / (max_level - min_level)
    return min_radius + t * (max_radius - min_radius)


def make_robot(state, level=1, behaviour='train'):
    radius = radius_from_level(level)
    return {
        'kinematics': {'name': 'diff'},
        'state': [state[0] + 0.5, state[1] + 0.5, state[2]],
        'shape': {'name': 'circle', 'radius': radius},
        'level': level,
        'behavior': {'name': f'{behaviour}'},
        'plot': {
            'show_trajectory': False,
            'show_goal': False
        },
        'unobstructed': True
    }


def create_apple(center=(2, 1), level=1):
    cx, cy = center[0] + 0.5, center[1] + 0.5
    radius = radius_from_level(level)

    return {
        'kinematics': {'name': 'diff'},
        'state': [cx, cy, 1],
        'shape': {'name': 'circle', 'radius': radius},
        'behavior': {'name': 'apple'},
        'color': 'red',
        'plot': {
            'show_trajectory': False,
            'show_goal': False
        },
        'unobstructed': True
    }


def generate_yaml(
        world_size=(10, 10),
        robots=None,
        apples=None,
        behaviour='train',
        output_file='grid.yaml'
):
    data = {
        'world': {'size': list(world_size)},
        'robot': [],
        'obstacle': []
    }

    for x in range(world_size[0] + 1):
        data['obstacle'].append({
            'shape': {'name': 'linestring', 'vertices': [[x, 0], [x, world_size[1]]]},
            'state': [0, 0, 0],
            'unobstructed': True
        })

    for y in range(world_size[1] + 1):
        data['obstacle'].append({
            'shape': {'name': 'linestring', 'vertices': [[0, y], [world_size[0], y]]},
            'state': [0, 0, 0],
            'unobstructed': True
        })

    for a in apples:
        center = a['state']
        level = a['level']
        data['robot'].append(create_apple(center=center, level=level))

    for r in robots:
        data['robot'].append(
            make_robot(
                state=r.get('state', (0, 0, 0)),
                level=r.get('level', 1),
                behaviour=behaviour
            )
        )

    with open(output_file, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        print(f'Wrote to {output_file}')
