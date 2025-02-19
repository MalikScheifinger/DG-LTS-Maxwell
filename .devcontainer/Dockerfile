FROM dealii/dealii:v9.5.0-jammy
USER root
RUN  echo 'dealii:dealii' | chpasswd
EXPOSE 22
RUN service ssh start
RUN ["apt-get", "-y", "update"]

# LLVM
RUN ["apt-get", "-y", "install", "clang"]
RUN ["apt-get", "-y", "install", "lldb"]
RUN ["apt-get", "-y", "install", "lld"]

# UTILS
RUN ["apt-get", "-y", "install", "doxygen"]
RUN ["apt-get", "-y", "install", "gdb"]
RUN ["apt-get", "-y", "install", "git"]

USER dealii
WORKDIR /home/dealii