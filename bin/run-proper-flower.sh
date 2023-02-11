#!/bin/bash

echo "Starting server"
# python server.py &
./term.scpt "cd $PWD && python server.py"
sleep 3 # Sleep for 3s to give the server enough time to start

number_of_clients=3
for ((n=0;n<$number_of_clients;n++)); do
    echo "Starting client $n"
    ./term.scpt "cd $PWD && python client.py -u $n -nu $number_of_clients"
    # python client.py -u $n -nu $number_of_clients &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait