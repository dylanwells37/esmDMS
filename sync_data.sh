#!/bin/bash
CLUSTER="dhw28@cluster.csb.pitt.edu"
REMOTE_PATH="/net/dali/home/barton/dhw28/popDMS/esmDMS/data"
LOCAL_PATH="/Users/dylanwells/CodingProjects/popDMS/esmDMS/data"

rsync -avz --progress --partial \
  --exclude="*.tmp" \
  $CLUSTER:$REMOTE_PATH $LOCAL_PATH