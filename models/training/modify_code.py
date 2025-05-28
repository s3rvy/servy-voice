PIPER_PATH: str = "./models/training/resources/piper-sample-generator"
OPENWAKEWORD_PATH: str = "./models/training/resources/openwakeword"

def modify_python_file(file_path, old_code, new_code):
    """
    Replaces occurrences of old_code with new_code in the file specified by file_path.

    Args:
        file_path (str): The path to the Python file to modify.
        old_code (str): The code to be replaced.
        new_code (str): The code to replace old_code with.
    """
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    modified_content = file_content.replace(old_code, new_code)

    with open(file_path, 'w') as file:
        file.write(modified_content)

if __name__ == "__main__":
    # Restore default behavior before pytorch 2.6
    piper_generate_samples_file_path = f"{PIPER_PATH}/generate_samples.py"
    load_with_weights_only_true = "torch.load(model_path)"
    load_with_weights_only_false = "torch_model = torch.load(model_path, weights_only=False)"

    modify_python_file(piper_generate_samples_file_path, load_with_weights_only_true, load_with_weights_only_false)

    # Disable multithreading in openwakeword train.py for data loading
    openwakeword_train_file_path = f"{OPENWAKEWORD_PATH}/openwakeword/train.py"
    data_loader_multithreading_config = "num_workers=n_cpus, prefetch_factor=16"

    modify_python_file(openwakeword_train_file_path, data_loader_multithreading_config, "")