# LLM_MedGemma documentation!

## Description

LLM project using MedGemma model with prompt engineering for tomographic image analysis.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://imgs/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://imgs/data/` to `data/`.


