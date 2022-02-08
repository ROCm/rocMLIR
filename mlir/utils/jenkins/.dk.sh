mkdir /home/swaters
UID=1010
GID=1010
addgroup --gid  swaters
adduser --system --home /home/swaters --no-create-home --uid 1010 --gid  swaters
usermod -a -G sudo swaters
apt update
apt install sudo
apt install lldb-10
apt install vim emacs
