import yaml


def dump(data, file_path):
    yaml_str = yaml.dump(data)

    with open(file_path, "w") as f:
        f.write(yaml_str)

def read(file_path):
    with open(file_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception(
                f"Error in reading {file_path} file: with exception->{exc}")

    return data

def get_global_config(file_path):
    global_data = read(file_path)
    return global_data
