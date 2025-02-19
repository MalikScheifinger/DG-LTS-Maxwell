# VSCode deal.ii Development Container

Provides a development container configuration for deal.ii.

For any information about the development inside a container, refer to the official VSCode [documentation](https://code.visualstudio.com/docs/devcontainers/containers).

# Forwarding SSH keys

In order to use the VSCode git plugin, one needs to forward his private key for gitlab.kit.edu identification with an ssh-agent.
Click [here](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials) for details.

The **best practice** is to start an ssh-agent with .bash_profile file.


1. Add the following to your ~/.bash_profile
    ```bash
    if [ -z "$SSH_AUTH_SOCK" ]; then
    # Check for a currently running instance of the agent
    RUNNING_AGENT="`ps -ax | grep 'ssh-agent -s' | grep -v grep | wc -l | tr -d '[:space:]'`"
    if [ "$RUNNING_AGENT" = "0" ]; then
            # Launch a new instance of the agent
            ssh-agent -s &> $HOME/.ssh/ssh-agent
    fi
    eval `cat $HOME/.ssh/ssh-agent`
    fi
    ```
2. Logout and login again
3. Add your private key to be managed with the ssh-agent
    ```bash
        ssh-add $PATH_TO_YOUR_PRIVATE_KEY
    ```
    E.g. `$PATH_TO_YOUR_PRIVATE_KEY = $HOME/.ssh/id_rsa`
