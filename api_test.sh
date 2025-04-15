#!/bin/bash
img_b64_str=$(base64 -i assets/demo.png)

curl -X POST "http://localhost:8082/generate" \
     -H "Content-Type: application/json" \
     -d '{ "image": "'"$img_b64_str"'", }' \
     -o test2.glb
