import json

DEFAULT_SETTINGS = {
    "3d_strength": 1.0,
    "maximum_disparity": 15,
    "depth_inversion": False,
    "temporal_smoothing": 0.7
}


def save_settings(settings, filename="settings.json"):
    with open(filename, "w") as file:
        json.dump(settings, file, indent=4)


def load_settings(filename="settings.json"):
    try:
        with open(filename, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        return DEFAULT_SETTINGS.copy()