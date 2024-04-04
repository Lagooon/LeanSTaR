#! /bin/bash

# assert $1 is not empty
if [ -z "$1" ]; then
    echo "Please specify a file to run."
    exit 1
fi

# assert $1 is a valid file
if [ ! -f "$1" ]; then
    echo "File $1 does not exist."
    exit 1
fi

nohup bash $1 >/tmp/output.out 2>&1 &

tail -f /tmp/output.out
