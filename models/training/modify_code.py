PIPER_PATH: str = "./resources/piper-sample-generator"
OPENWAKEWORD_PATH: str = "./resources/openwakeword"

def modify_python_file(file_path, old_code, new_code) -> None:
    """
    Replaces occurrences of old_code with new_code in the file specified by file_path.

    Args:
        file_path (str): The path to the Python file to modify.
        old_code (str): The code to be replaced.
        new_code (str): The code to replace old_code with.
    """
    try:
        with open(file_path, 'r') as file:
            file_content: str = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    modified_content:str = file_content.replace(old_code, new_code)

    with open(file_path, 'w') as file:
        file.write(modified_content)

if __name__ == "__main__":
    # Disable multithreading in openwakeword train.py for data loading
    openwakeword_train_file_path: str = f"{OPENWAKEWORD_PATH}/openwakeword/train.py"
    data_loader_multithreading_config = "num_workers=n_cpus, prefetch_factor=16"

    modify_python_file(openwakeword_train_file_path, data_loader_multithreading_config, "")

    # Modify the openwakeword train.py file to remove ONNX to TFLite conversion after training
    code: str = """convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),
                               os.path.join(config["output_dir"], config["model_name"] + ".tflite"))"""
    modify_python_file(openwakeword_train_file_path, code, "")