#!/bin/sh
cd /data/hotspot-featurextract-service
LOGS_DIR=logs

if [ ! -d "${LOGS_DIR}" ]
then
  mkdir "${LOGS_DIR}"
fi

python3 hotspot_featurextract_service.py hotspot_featurextract_service.conf

echo "hotspot_featurextract_service.py starting..."
