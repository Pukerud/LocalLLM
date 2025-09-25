#!/bin/bash
set -e # Exit immediately if any command fails

# --- Part 1: Configuration ---
# Use $HOME for portability
INSTALL_DIR="$HOME/lm-studio-server"
MODEL_DIR="$HOME/LLM"
SILLY_TAVERN_DIR="$HOME/sillytavern"
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_FILENAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Part 2: Directory and Model Preparation (skipped on update) ---
if [ "$1" != "--update" ]; then
    echo "--- [Step 1/7] Preparing directories and default model... ---"
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$MODEL_DIR"
    mkdir -p "$SILLY_TAVERN_DIR"
    if [ ! -f "$MODEL_DIR/$MODEL_FILENAME" ]; then
        echo "Default model not found. Downloading..."
        wget -O "$MODEL_DIR/$MODEL_FILENAME" "$MODEL_URL"
    fi
else
    echo "--- [Update Mode] Skipping directory and model preparation. ---"
fi

# --- Part 3: Create the Initial SillyTavern Configuration ---
echo ""
echo "--- [Step 2/7] Creating the initial SillyTavern configuration... ---"
cat <<EOF > "${SILLY_TAVERN_DIR}/config.yaml"
# This configuration file enables whitelist mode for security.
# To allow access from other machines, add their IP addresses here.
whitelist:
  - "127.0.0.1"
EOF
chmod 777 "${SILLY_TAVERN_DIR}/config.yaml"
echo "Initial configuration created."

# --- Part 4: Create the Unified Docker Compose File from Template ---
echo ""
echo "--- [Step 3/7] Creating the unified server and UI configuration... ---"

# Create a template for the docker-compose file with placeholders
cat <<EOF > "${INSTALL_DIR}/docker-compose.template.yml"
version: '3.8'

services:
  llm-api:
    image: nvidia/cuda:12.1.1-devel-ubuntu22.04
    container_name: llm-api-server
    restart: unless-stopped
    network_mode: "host"
    volumes:
      - ##MODEL_DIR##:/models
      - llama_cpp_pip_cache:/pip_cache
    entrypoint: /models/build_and_run.sh
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  sillytavern:
    image: ghcr.io/sillytavern/sillytavern:latest
    container_name: sillytavern
    restart: unless-stopped
    network_mode: "host"
    volumes:
      - ##SILLY_TAVERN_DIR##/config.yaml:/home/node/app/config.yaml:rw
      - ##SILLY_TAVERN_DIR##/data:/home/node/app/data:rw
      - ##SILLY_TAVERN_DIR##/extensions:/home/node/app/public/scripts/extensions/third-party:rw
      - ##SILLY_TAVERN_DIR##/plugins:/home/node/app/plugins:rw
    command: ["node", "server.js", "--port", "8001"]
    depends_on:
      - llm-api

volumes:
  llama_cpp_pip_cache:
EOF

# Use sed to replace placeholders with actual paths, handling potential special characters
# This creates the final, definitive docker-compose.yml
sed -e "s|##MODEL_DIR##|${MODEL_DIR}|g" \
    -e "s|##SILLY_TAVERN_DIR##|${SILLY_TAVERN_DIR}|g" \
    "${INSTALL_DIR}/docker-compose.template.yml" > "${INSTALL_DIR}/docker-compose.yml"

echo "Docker Compose file created successfully."

# --- Part 5: Create the Build-and-Run Script for the LLM Server ---
echo ""
echo "--- [Step 4/7] Creating the build-from-source script... ---"
cat <<EOF > "${MODEL_DIR}/build_and_run.sh"
#!/bin/bash
set -e
export PIP_CACHE_DIR=/pip_cache
apt-get update && apt-get install -y --no-install-recommends python3 python3-pip build-essential g++ cmake ninja-build git
if ! pip show uvicorn > /dev/null 2>&1; then
  echo "Building llama-cpp-python from source..."
  export CMAKE_ARGS="-DGGML_CUDA=on"
  export FORCE_CMAKE=1
  pip install 'llama-cpp-python[server]'
fi
echo "Starting the LLM API server..."
exec python3 -m llama_cpp.server --model /models/${MODEL_FILENAME} --n_gpu_layers -1 --host 0.0.0.0 --port 8000
EOF
chmod +x "${MODEL_DIR}/build_and_run.sh"

# --- Part 6: Create the Final Management Script (v2.1 - The Definitive Edition) ---
echo ""
echo "--- [Step 5/7] Creating the LLM Server Management Script (llm_manager.sh)... ---"
cat <<EOF > "${INSTALL_DIR}/llm_manager.sh"
#!/bin/bash
# --- Configuration (using absolute paths for reliability) ---
SCRIPT_DIR="${INSTALL_DIR}"
CONFIG_FILE="${MODEL_DIR}/build_and_run.sh"
MODEL_DIR="${MODEL_DIR}"
SILLY_TAVERN_CONFIG_FILE="${SILLY_TAVERN_DIR}/config.yaml"

# --- Colors for better UI ---
GREEN=\$(tput setaf 2); YELLOW=\$(tput setaf 3); BLUE=\$(tput setaf 4); RESET=\$(tput sgr0)

# --- Function to reliably restart all services ---
function restart_all_services() { echo "Stopping all services..."; docker compose --project-directory "\$SCRIPT_DIR" down; echo "Starting all services..."; docker compose --project-directory "\$SCRIPT_DIR" up -d; }

# --- Menu Functions ---
function search_models() { echo ""; echo "\${BLUE}Find models at: \${YELLOW}https://huggingface.co/TheBloke\${RESET}"; echo ""; read -p "Press [Enter] to continue..."; }
function download_model() { echo ""; read -p "Paste GGUF URL: " u; if [[ -z "\$u" ]]; then echo "\${YELLOW}No URL provided. Aborting.\${RESET}"; return; fi; c="\${u%%\\?*}"; f=\$(basename "\$c"); echo "\${BLUE}Downloading to: '\${f}'...\${RESET}"; wget -O "\${MODEL_DIR}/\${f}" "\$u"; if [ \$? -eq 0 ]; then echo "\${GREEN}Download successful!\${RESET}"; else echo "\${YELLOW}Download failed.\${RESET}"; fi; read -p "Press [Enter] to continue..."; }
function switch_model() { echo ""; echo "\${BLUE}Scanning for models... \${RESET}"; mapfile -t m < <(find "\$MODEL_DIR" -maxdepth 1 -type f -name "*.gguf" -printf "%f\n"); if [ \${#m[@]} -eq 0 ]; then echo "\${YELLOW}No models found in \${MODEL_DIR}.\${RESET}"; read -p "Press [Enter] to continue..."; return; fi; echo "\${YELLOW}Select a model to use:\${RESET}"; PS3="Your choice: "; select mc in "\${m[@]}"; do if [[ -n "\$mc" ]]; then echo "\${BLUE}Selected model: \${mc}\${RESET}"; echo "Updating configuration..."; sed -i "s|--model /models/.*\\.gguf|--model /models/\${mc}|" "\$CONFIG_FILE"; echo "Restarting services..."; restart_all_services; echo "\${GREEN}Services are restarting with the new model: \${mc}!\${RESET}"; break; else echo "Invalid selection."; fi; done; read -p "Press [Enter] to continue..."; }
function set_context_window() { echo ""; CUR_C=\$(grep -o '\\--n_ctx [0-9]*' "\$CONFIG_FILE" | awk '{print \$2}'); if [[ -z "\$CUR_C" ]]; then echo "\${BLUE}Current context size (n_ctx): \${GREEN}Default\${RESET}"; else echo "\${BLUE}Current context size (n_ctx): \${GREEN}\${CUR_C}\${RESET}"; fi; echo ""; echo "\${YELLOW}Select a new context size:\${RESET}"; opts=("2048" "4096" "8192" "16384" "Custom" "Remove"); PS3="Your choice: "; select ch in "\${opts[@]}"; do case "\$ch" in "Custom") read -p "Enter custom size: " N_C; if ! [[ "\$N_C" =~ ^[0-9]+\$ ]]; then echo "Invalid number."; return; fi; break;; *) N_C=\${ch%% *}; break;; esac; done; if [[ "\$N_C" == "Remove" ]]; then echo "\${BLUE}Removing n_ctx setting...\${RESET}"; sed -i 's/ --n_ctx [0-9]*//' "\$CONFIG_FILE"; else echo "\${BLUE}Setting n_ctx to \${N_C}...\${RESET}"; if grep -q "\\--n_ctx" "\$CONFIG_FILE"; then sed -i "s|--n_ctx [0-9]*|--n_ctx \${N_C}|" "\$CONFIG_FILE"; else sed -i "s|exec python3.*|& --n_ctx \${N_C}|" "\$CONFIG_FILE"; fi; fi; echo "Restarting services..."; restart_all_services; echo "\${GREEN}Services are restarting with the new context size!\${RESET}"; read -p "Press [Enter] to continue..."; }
function manage_whitelist() { echo ""; echo "\${BLUE}--- Current Whitelist --- \${RESET}"; grep -v '^#' "\$SILLY_TAVERN_CONFIG_FILE" | grep -v '^\s*\$'; echo ""; read -p "Enter the IP address to add: " N_IP; if ! [[ "\$N_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then echo "\${YELLOW}Invalid IP address format.\${RESET}"; read -p "Press [Enter] to continue..."; return; fi; if grep -q "\${N_IP}" "\$SILLY_TAVERN_CONFIG_FILE"; then echo "\${YELLOW}IP address already in whitelist.\${RESET}"; else echo "\${BLUE}Adding \${N_IP} to whitelist...\${RESET}"; echo "  - \"\${N_IP}\"" >> "\$SILLY_TAVERN_CONFIG_FILE"; echo "\${GREEN}IP address added.\${RESET}"; fi; echo "Forcing recreation of SillyTavern container to apply changes..."; docker compose --project-directory "\$SCRIPT_DIR" up -d --force-recreate sillytavern; echo "\${GREEN}SillyTavern restarted successfully!\${RESET}"; read -p "Press [Enter] to continue..."; }
function view_llm_log() { clear; echo "\${BLUE}--- Live Logs: LLM API Server --- \${RESET}"; echo "\${YELLOW}Press [Ctrl+C] to return to the menu.\${RESET}"; docker logs -f llm-api-server; read -p "Press [Enter] to continue..."; }
function view_sillytavern_log() { clear; echo "\${BLUE}--- Live Logs: SillyTavern UI --- \${RESET}"; echo "\${YELLOW}Press [Ctrl+C] to return to the menu.\${RESET}"; docker logs -f sillytavern; read -p "Press [Enter] to continue..."; }
function check_status() { echo ""; echo "\${BLUE}--- Docker Container Status --- \${RESET}"; docker ps; echo ""; echo "\${BLUE}--- Active Configuration --- \${RESET}"; CUR_M=\$(grep '\\--model /models/' "\$CONFIG_FILE" | sed -E 's|.*--model /models/([^ ]+).*|\\1|'); if [[ -n "\$CUR_M" ]]; then echo "Model: \${GREEN}\${CUR_M}\${RESET}"; else echo "\${YELLOW}No model configured.\${RESET}"; fi; CUR_C=\$(grep -o '\\--n_ctx [0-9]*' "\$CONFIG_FILE" | awk '{print \$2}'); if [[ -z "\$CUR_C" ]]; then echo "Context: \${YELLOW}Default\${RESET}"; else echo "Context: \${GREEN}\${CUR_C}\${RESET}"; fi; echo ""; read -p "Press [Enter] to continue..."; }
function toggle_whitelist() {
    echo ""
    if grep -q "^#whitelist:" "\$SILLY_TAVERN_CONFIG_FILE"; then
        echo "\${BLUE}Whitelist is currently disabled. Enabling it...\${RESET}"
        sed -i 's/^#whitelist:/whitelist:/' "\$SILLY_TAVERN_CONFIG_FILE"
        echo "\${GREEN}Whitelist enabled.\${RESET}"
    elif grep -q "^whitelist:" "\$SILLY_TAVERN_CONFIG_FILE"; then
        echo "\${BLUE}Whitelist is currently enabled. Disabling it...\${RESET}"
        sed -i 's/^whitelist:/#whitelist:/' "\$SILLY_TAVERN_CONFIG_FILE"
        echo "\${GREEN}Whitelist disabled.\${RESET}"
    else
        echo "\${YELLOW}Could not determine whitelist status. No changes made.\${RESET}"
        read -p "Press [Enter] to continue..."
        return
    fi
    echo "Forcing recreation of SillyTavern container to apply changes..."
    docker compose --project-directory "\$SCRIPT_DIR" up -d --force-recreate sillytavern
    echo "\${GREEN}SillyTavern restarted successfully!\${RESET}"
    read -p "Press [Enter] to continue..."
}
function update_script() {
    echo ""
    echo "\${BLUE}Checking for updates... \${RESET}"
    local update_url="https://raw.githubusercontent.com/Pukerud/LocalLLM/main/setup.sh"

    echo "Downloading latest version from \${YELLOW}\$update_url\${RESET}..."
    local temp_script="/tmp/setup_latest.sh"

    # Use curl with -f to fail silently on server errors (like 404)
    if curl -fsSL -o "\$temp_script" "\$update_url"; then
        # Check if the downloaded file is a valid shell script and not a GitHub 404 page
        if head -n 1 "\$temp_script" | grep -q "^#\!/bin/bash"; then
            echo "\${GREEN}Download complete. Applying update...\${RESET}"
            chmod +x "\$temp_script"
            # Execute the new setup script with the --update flag and exit the manager
            exec bash "\$temp_script" --update
        else
            echo "\${YELLOW}Downloaded file is not a valid script. The URL may be incorrect.\${RESET}"
            rm -f "\$temp_script"
            read -p "Press [Enter] to continue..."
        fi
    else
        echo "\${YELLOW}Failed to download the update. Please check your internet connection or the repository URL.\${RESET}"
        read -p "Press [Enter] to continue..."
    fi
}

# --- Main Menu Loop ---
while true; do
    clear
    echo "\${GREEN}================== LLM Server & Model Manager ==================\${RESET}"
    echo "Select an option:"
    echo "\${YELLOW}1) \${RESET} Search for Models"
    echo "\${YELLOW}2) \${RESET} Download a New Model"
    echo "\${YELLOW}3) \${RESET} Switch Active Model"
    echo "\${YELLOW}4) \${RESET} Set LLM Context Size"
    echo "\${YELLOW}5) \${RESET} Manage UI Whitelist"
    echo "\${YELLOW}6) \${RESET} View LLM API Log"
    echo "\${YELLOW}7) \${RESET} View SillyTavern UI Log"
    echo "\${YELLOW}8) \${RESET} Check Service Status"
    echo "\${YELLOW}9) \${RESET} Toggle Whitelist (ON/OFF)"
    echo "\${YELLOW}10) \${RESET} Update from GitHub"
    echo "\${YELLOW}11) \${RESET} Exit"
    echo ""
    read -p "Enter your choice [1-11]: " C
    case "\$C" in
        1) search_models;;
        2) download_model;;
        3) switch_model;;
        4) set_context_window;;
        5) manage_whitelist;;
        6) view_llm_log;;
        7) view_sillytavern_log;;
        8) check_status;;
        9) toggle_whitelist;;
        10) update_script;;
        11) echo "Exiting."; break;;
        *) echo "Invalid option. Please try again."; read -p "Press [Enter] to continue...";;
    esac
done
EOF
chmod +x "${INSTALL_DIR}/llm_manager.sh"
echo "Management script created successfully."

# --- Part 7: Launch the Entire Stack ---
echo ""
echo "--- [Step 6/7] Stopping any old services... ---"
docker compose --project-directory "$INSTALL_DIR" down --remove-orphans

echo ""
echo "--- [Step 7/7] Launching the complete stack... ---"
echo "--- The first launch will take 5-10 MINUTES to compile the engine. Please be patient. ---"
docker compose --project-directory "$INSTALL_DIR" up -d

echo ""
echo "--- 🎉🎉🎉 DEPLOYMENT COMPLETE 🎉🎉🎉 ---"
echo "Wait 5-10 minutes for the one-time compilation. You can monitor the progress with: docker logs -f llm-api-server"
echo "  - 🧠 Your LLM API is at: http://YOUR_SERVER_IP:8000/docs"
echo "  - 💬 Your Chat Interface is at: http://YOUR_SERVER_IP:8001"
echo ""
echo "IMPORTANT: To access the UI from another machine, you must add its IP address to the whitelist."
echo "Run the manager script to add your IP:"
echo "${GREEN}cd ${INSTALL_DIR} && ./llm_manager.sh${RESET}"