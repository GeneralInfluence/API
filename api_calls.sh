#!/bin/bash

# GET REQUESTS
curl -i -H

# POST REQUESTS POST REQUESTS POST REQUESTS POST REQUESTS POST REQUESTS
# Initialize
curl -X POST -H "Content-Type: application/json" -d '{"models":["untrustworthy"]}' "http://0.0.0.0:6969/v1/gizmoduck/initialize/?app_id=SYNAPSIFY&app_key=908cecfa-9944-4e67-a05a-f9c200a9533e"
curl -X POST -H "Content-Type: application/json" -d '{"models":["untrustworthy"]}' "http://52.21.52.28:9999/v1/gizmoduck/initialize/?app_id=SYNAPSIFY&app_key=908cecfa-9944-4e67-a05a-f9c200a9533e"
# Classify
curl -X POST -H "Content-Type: application/json" -d '{"sentences":["what it is mutha fucka"],"model":["untrustworthy"]}' "http://0.0.0.0:9999/v1/gizmoduck/classify_one/?app_id=SYNAPSIFY&app_key=908cecfa-9944-4e67-a05a-f9c200a9533e"