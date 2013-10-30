#! /bin/bash

TRAINING_FILE=$1
BASENAME=${TRAINING_FILE%.*}
shift
TEST_FILE=$1
shift

./main.py $TRAINING_FILE -o $BASENAME.model $* > /dev/null
./evaluate.py $BASENAME.model $TEST_FILE



