#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 ../az-run-task.sh -c config/config-transformer.yaml &> az-transformer.out &