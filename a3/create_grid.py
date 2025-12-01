import yaml


def make_robot(state, level=1, radius=0.2):
    return {
        "kinematics": {"name": "diff"},
        "state": [state[0] + 0.5, state[1] + 0.5, state[2]],
        "shape": {"name": "circle", "radius": radius},
        "level": level,
        "behavior": {"name": "train"},
        "plot": {
            "show_trajectory": False,
            "show_goal": False
        },
        "unobstructed": True
    }


def create_apple(center=(2, 1), radius=0.25):
    cx, cy = center[0] + 0.5, center[1] + 0.5

    return [
        {
            "shape": {"name": "circle", "radius": radius},
            "state": [cx, cy, 0],
            "color": "red",
            "unobstructed": True
        },
        {
            "shape": {"name": "circle", "radius": radius * 0.28},
            "state": [cx + radius * 0.5, cy + radius * 0.7, 0],
            "color": "green",
            "unobstructed": True
        }
    ]


def generate_yaml(
        world_size=(10, 10),
        robots=None,
        apples=None,
        output_file="grid.yaml"
):
    data = {
        "world": {"size": list(world_size)},
        "robot": [],
        "obstacle": []
    }

    for x in range(world_size[0] + 1):
        data["obstacle"].append({
            "shape": {"name": "linestring", "vertices": [[x, 0], [x, world_size[1]]]},
            "state": [0, 0, 0],
            "unobstructed": True
        })

    for y in range(world_size[1] + 1):
        data["obstacle"].append({
            "shape": {"name": "linestring", "vertices": [[0, y], [world_size[0], y]]},
            "state": [0, 0, 0],
            "unobstructed": True
        })

    for r in robots:
        data["robot"].append(
            make_robot(
                state=r.get("state", (0, 0, 0)),
                level=r.get("level", 1),
                radius=r.get("radius", 0.2),
            )
        )

    for a in apples:
        for item in create_apple(a):
            data["obstacle"].append(item)

    with open(output_file, "w") as f:
        yaml.dump(data, f, sort_keys=False)
        print(f'Wrote to {output_file}')
