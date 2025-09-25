# Local LLM Server & UI Automation

This project provides a set of scripts to completely automate the setup and management of a local Large Language Model (LLM) environment using Docker. It includes a GPU-accelerated API server for the LLM and the SillyTavern web UI for chat.

The main goal is to provide a simple, one-command setup and an easy-to-use menu for managing the environment afterward.

## Features

- **Automated Setup**: A single `setup.sh` script prepares directories, downloads a default model, and configures all necessary files.
- **Dockerized Services**: The entire stack (LLM API server and SillyTavern UI) runs in isolated Docker containers.
- **GPU Acceleration**: The LLM server is configured to use NVIDIA GPUs via `llama-cpp-python` for high-performance inference.
- **Interactive Manager**: A powerful `llm_manager.sh` script provides a menu for:
    - Searching for new models on Hugging Face.
    - Downloading new models (GGUF format).
    - Switching the active LLM on the fly.
    - Adjusting the LLM's context window size.
    - Managing an IP whitelist for accessing the UI.
    - Viewing live logs for both services.
    - Checking the status of the services.
    - Updating the entire script suite from GitHub.
- **Portable**: Uses the user's home directory (`$HOME`) for all file paths, avoiding hardcoded paths.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Docker**: The containerization platform used to run the services.
    - Install Docker: `https://docs.docker.com/engine/install/`
2.  **NVIDIA GPU Drivers**: Required for GPU acceleration.
3.  **NVIDIA Container Toolkit**: Allows Docker containers to access the GPU.
    - Install Toolkit: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`
4.  **`curl` and `wget`**: Used by the scripts to download the installer and models. (e.g., `sudo apt-get install curl wget`)
5.  **`tput`**: Used for colored output in the manager script (usually included in `ncurses`).

## How to Use

### 1. Quick Install (One-Liner)

For a fast and easy setup, you can run the following command in your terminal. This will download the `setup.sh` script and execute it automatically.

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Pukerud/LocalLLM/main/setup.sh)"
```

### 2. Manual Setup

If you prefer, you can download the `setup.sh` script first and then run it manually.
```bash
wget https://raw.githubusercontent.com/your-username/your-repo/main/setup.sh
chmod +x setup.sh
./setup.sh
```

The script will perform the following actions:
1.  Create three directories in your home folder:
    - `~/lm-studio-server`: Contains the `docker-compose.yml` and the manager script.
    - `~/LLM`: Stores your GGUF model files.
    - `~/sillytavern`: Stores SillyTavern data, configs, and extensions.
2.  Download a default model (`Mistral-7B-Instruct-v0.2`) to get you started.
3.  Generate the `docker-compose.yml` file.
4.  Generate the `build_and_run.sh` script for the LLM container.
5.  Generate the `llm_manager.sh` script for easy management.
6.  Start the Docker containers.

**Important**: The very first time you run this, the `llm-api-server` container will take **5-10 minutes** to build the `llama-cpp-python` engine from source with CUDA support. This is a one-time process. You can monitor the progress with:

```bash
docker logs -f llm-api-server
```

Once the build is complete, your services will be available:
- **LLM API Server**: `http://<YOUR_SERVER_IP>:8000/docs`
- **SillyTavern UI**: `http://<YOUR_SERVER_IP>:8001`

### 2. Managing the Environment

After the initial setup, you can manage your entire LLM environment using the `llm_manager.sh` script.

First, navigate to the installation directory:
```bash
cd ~/lm-studio-server
```

Then, run the manager:
```bash
./llm_manager.sh
```

This will open an interactive menu with the following options:

- **Search for Models**: Opens a link to TheBloke's Hugging Face page, a great source for GGUF models.
- **Download a New Model**: Prompts you to paste a URL to a GGUF model file, which it will download into your `~/LLM` directory.
- **Switch Active Model**: Lists all `.gguf` files in your `~/LLM` directory and allows you to select one. It will automatically update the configuration and restart the services.
- **Set LLM Context Size**: Allows you to set or change the `n_ctx` (context window size) parameter for the LLM and restarts the services.
- **Manage UI Whitelist**: By default, SillyTavern is only accessible from `127.0.0.1`. Use this option to add the IP address of other machines on your network so you can access the UI from them.
- **View Logs**: Tails the live logs for either the LLM API or the SillyTavern UI, which is useful for debugging.
- **Check Service Status**: Shows the status of your Docker containers and the currently configured model and context size.

This setup provides a robust and flexible way to run and manage your own local LLM services. Enjoy!
