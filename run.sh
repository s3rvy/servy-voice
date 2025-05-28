#!/bin/bash

set -e
VENV_DIR=".venv"

ensure_venv() {
    if [ ! -f ".python-version" ]; then
        echo ".python-version file not found. Please create one with the desired Python version."
        exit 1
    fi

    PYTHON_VERSION=$(pyenv version | awk '{print $1}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?')
    echo $PYTHON_VERSION

    if ! command -v pyenv >/dev/null 2>&1; then
        echo "pyenv is not installed. Please install pyenv first."
        exit 1
    fi

    # Find the correct python executable from pyenv
    PYENV_PYTHON="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"

    if [ ! -x "$PYENV_PYTHON" ]; then
        pyenv install -s "$PYTHON_VERSION"
    fi

    if [ -d "$VENV_DIR" ]; then
        # Use the venv's python to check version
        ACTIVE_VERSION=$("$VENV_DIR/bin/python" -c "import platform; print(platform.python_version())")
        if [[ "$ACTIVE_VERSION" == "$PYTHON_VERSION"* ]]; then
            "${VENV_DIR}/bin/pip" install --upgrade pip
            # Activate and return
            return
        else
            echo "Existing venv Python version ($ACTIVE_VERSION) does not match required version ($PYTHON_VERSION). Recreating venv."
            rm -rf "$VENV_DIR"
        fi
    fi

    # Create new venv with correct python version
    "$PYENV_PYTHON" -m venv "$VENV_DIR"
    "${VENV_DIR}/bin/pip" install --upgrade pip
}

train_activation_model() {
    ensure_venv

    RESOURCE_PATH="./models/training/resources"
    PIPER_PATH="$RESOURCE_PATH/piper-sample-generator"
    OPENWAKEWORD_PATH="$RESOURCE_PATH/openwakeword"
    OPENWAKEWORD_MODEL_PATH="${OPENWAKEWORD_PATH}/openwakeword/resources/models"
    OTHER_MODELS_PATH="${RESOURCE_PATH}/other_models"
    TRAIN_MODEL_PATH="${RESOURCE_PATH}/hey_servy_model"

    echo "--------------------------------------- Setting up environment ---------------------------------------"

    "${VENV_DIR}/bin/pip" install -r ./models/training/requirements.txt

    if [ ! -x "$RESOURCE_PATH" ]; then
        mkdir ./models/training/resources
        mkdir ./models/training/resources/data
    fi

    if [ ! -x "$PIPER_PATH" ]; then
        git clone https://github.com/rhasspy/piper-sample-generator "${PIPER_PATH}"
        wget -O "${PIPER_PATH}/models/en_US-libritts_r-medium.pt" 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
    fi

    if [ ! -x "$OPENWAKEWORD_PATH" ]; then
        git clone https://github.com/dscripka/openwakeword "${OPENWAKEWORD_PATH}"
    fi
    "${VENV_DIR}/bin/pip" install -e "${OPENWAKEWORD_PATH}"

    if [ ! -x "$OPENWAKEWORD_MODEL_PATH" ]; then
        mkdir -p "${OPENWAKEWORD_MODEL_PATH}"

        wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O "${OPENWAKEWORD_MODEL_PATH}/embedding_model.onnx"
        wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O "${OPENWAKEWORD_MODEL_PATH}/embedding_model.tflite"
        wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O "${OPENWAKEWORD_MODEL_PATH}//melspectrogram.onnx"
        wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O "${OPENWAKEWORD_MODEL_PATH}//melspectrogram.tflite"
    fi

    if [ ! -x "${OTHER_MODELS_PATH}" ]; then
        mkdir -p "${OTHER_MODELS_PATH}"
        # training set (~2,000 hours from the ACAV100M Dataset)
        # See https://huggingface.co/datasets/davidscripka/openwakeword_features for more information
        wget -O "${OTHER_MODELS_PATH}/feature_data.npy" https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy

        # validation set for false positive rate estimation (~11 hours)
        wget -O "${OTHER_MODELS_PATH}/false_positive_data.npy" https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
    fi

    "${VENV_DIR}/bin/python" ./models/training/modify_code.py

    echo "--------------------------------------- Setting up training data ---------------------------------------"

    # generate training data from datasets
    "${VENV_DIR}/bin/python" ./models/training/generate_training_data.py

    cd ./models/training/resources
    # generate synthetic training data
    "../../../${VENV_DIR}/bin/python" ./openwakeword/openwakeword/train.py --training_config ../servy_model.yml --generate_clips
    "../../../${VENV_DIR}/bin/python" ./openwakeword/openwakeword/train.py --training_config ../servy_model.yml --augment_clips

    echo "--------------------------------------- Train model ---------------------------------------"

    # train model
    "../../../${VENV_DIR}/bin/python" ./openwakeword/openwakeword/train.py --training_config ../servy_model.yml --train_model

#    rm -rf ./models/training/resources
}

print_help() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Available commands:"
    echo "  train_activation_model   Set up environment and train a custom OpenWakeWord activation model"
}

case "$1" in
    train_activation_model)
        train_activation_model
        ;;
    *)
        print_help
        exit 1
        ;;
esac