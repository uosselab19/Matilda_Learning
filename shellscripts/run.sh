#!/bin/sh

pid_search=$(netstat -nap | grep 8000 | awk '{print $7}')
pid_value=${#pid_search}

if [ $pid_value == 0 ]; then
    echo "Process already terminated."
else
    echo "Start terminate process"
    pid_val=${pid_search%%'/'*}
    kill -9 $pid_val
    if [ $? == 0 ]; then
        echo 'Terminate process success'
    else
        echo 'Terminate process fail'
    fi
fi

echo 'Process start'
nohup uvicorn main:app --reload --host 0.0.0.0 >> nohup.out 2>&1 &