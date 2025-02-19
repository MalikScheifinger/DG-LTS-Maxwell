# Using `enroot` with deal.II

Nvidia's `enroot` allows to convert privileged docker containers into unprivileged ones that can run in environments without root permissions, e.g. PDE cluster, BwUniCluster, ...

The following steps are necessary for building and converting a container:

1. Create docker image
    ```bash
        docker build -t dealii_stack .
    ```
2. Import docker image with enroot
    ```bash
        enroot import dockerd://dealii_stack
    ```
    This should produce the file `dealii_stack.sqsh` in the current directory.
    Note that you can also import images directly from Docker Hub. Details can be found [here](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md).
3. Create enroot image form sqsh file
    ```bash
        enroot create dealii_stack.sqsh
    ```
    Note that `-f` overrides images already created with the name dealii_stack.
4. Start enroot container
    ```bash
        enroot start -m .:/mnt dealii_stack
        cd /mnt
    ```
    This starts the container and mounts the current directory `.` to `/mnt` in the container file tree. Changes are persistent!