#!/bin/bash

# WSL2 exposes the Windows driver's libcuda under /usr/lib/wsl/lib; a native
# stub in /usr/lib/x86_64-linux-gnu can shadow it, so prefer the WSL copy.
if [ -d /usr/lib/wsl/lib ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

./assignment.exe "$@"
