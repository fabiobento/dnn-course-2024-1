#FROM tensorflow/tensorflow:2.14.0-gpu-jupyter
FROM tensorflow/tensorflow:2.14.0-gpu

# Evitar warnings ao partir para modo não interativo
ENV DEBIAN_FRONTEND=noninteractive
# Esse Dockerfile adiciona usuário "não-root" com acesso sudo. Para utilizá-lo Basta
# utilizar a propriedade "remoteUser" em devcontainer.json. No Linux a GID/UIDs do usuário 
# será atualizada para combinar com seu UID/GID local(quando utilizando a propriedade
# dockerFile).
# Veja https://aka.ms/vscode-remote/containers/non-root-user para detalhes.
ARG USERNAME=fabio
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Instalar as dependencias
COPY requirements.txt /
RUN pip install -r /requirements.txt
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    #   -------------    
    #   CONFIGURAÇÕES
    #   -------------    
# Criar usuário não-root
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Histórico persistente do bash history
    && SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo $SNIPPET >> "/root/.bashrc" \
    # [Optional] If you have a non-root user
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc" \
    #   -----------    
    #   INSTALAÇÕES
    #   -----------
    # Atualizar para preparar-se para instalações        
    && apt-get update \
    # Utilitários
    && apt-get -y install --no-install-recommends wget \    
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    # Instalar bibliotecas necessárias para o opencv
    && apt-get -y install libsm6 libxext6 libxrender-dev libgl1-mesa-glx\
    # Ferramenta para visualização do modelo
    && apt-get -y install graphviz\
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git iproute2 procps lsb-release \
    #   SQL Lite para armazenar trials de ajuste de hiperparâmetros (i.e. optuna)
    && apt-get -y install sqlite3 \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
