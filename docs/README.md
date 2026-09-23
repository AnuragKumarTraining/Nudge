# Nudge

Nudge is an event-driven computer vision pipeline designed for hospitality spaces like cafes and homestays. The system is built to monitor visual conditions, compare real-world captures against approved baselines or rules, and trigger downstream actions when a scene deviates from expected standards.

This repository contains the core pipeline logic, configuration, supporting utilities, and architecture documentation for the end-to-end application.

## Overview

At a high level, the project follows a pipeline-based design:

1. Images are collected from the environment or supplied as inputs.
2. A detection and feature extraction stage identifies relevant visual information.
3. Baselines and rules are applied to determine whether the current image matches the expected standard.
4. Results are stored, processed, and exposed to downstream services or clients.
5. The system can be extended to support event-driven reporting, approvals, and operational workflows.

## Architecture summary

The solution is organized around a modular event-driven architecture:

- Data acquisition: image feeds and capture sources
- Rule configuration: room, asset, and image rule definitions
- Feature extraction: computer vision logic that extracts meaningful scene features
- Baseline generation: reference images and structured comparison metadata
- Inference engine: evaluates current images against configured baselines
- Persistence layer: database models and migrations for storing results and metadata
- Workflow orchestration: Master/worker pipeline and helper modules for processing tasks

## Repository structure

```text
Nudge/
├── baselines/                # generated baseline references and metadata
├── config/                   # pipeline configuration files
├── current_images/            # current or incoming images
├── db/                       # database-related code or schemas
├── docs/                     # architecture documents and design notes
│   ├── backend-architecture-technical-spec_v1.1.md
│   ├── frontend-architecture-v3.md
│   ├── DB_diagram.pdf
│   └── README.md
├── image_rules/              # image/rule definitions and mappings
├── master_images/            # source master images used for baseline creation
├── migrations/               # database migration files
├── pipeline/                 # primary pipeline implementation
├── pipeline_helper/          # helper components for pipeline execution
├── prompts/                  # prompt or instruction definitions
├── utils/                    # reusable utilities for model loading and feature extraction
├── inference.py              # inference entry point
├── master.py                 # baseline generation script
├── schema.dbml               # database schema definition
├── .gitignore
└── README.md                 # this file
```

## Core workflow

### 1. Generate baselines

The project includes a baseline generation step via `master.py`.

- Reads master images from the configured master image directory
- Loads the YOLO model
- Extracts features from each image
- Saves a reference image and matching JSON metadata in `baselines/`

### 2. Run inference

The repository includes `inference.py`, which is intended to process live or incoming image data against the baseline definitions and rules.

### 3. Evaluate against configured rules

The system uses image rules and prompt-driven instructions to determine whether a capture matches the desired state. This forms the decision-making layer of the CV pipeline.

## Important design notes

- The repository is Python-based.
- The project uses a structured, modular pipeline approach instead of a single monolithic script.
- Architecture and backend specifications are documented under the `docs/` directory.
- The design is intended to support real-world hospitality operations such as compliance checks, room readiness validation, and timely action triggers.

## Documentation

For deeper technical design and system behavior, see:

- `docs/backend-architecture-technical-spec_v1.1.md`
- `docs/frontend-architecture-v3.md`
- `schema.dbml`

## Getting started

This project is structured for local development and experimentation in Python. Typical setup includes:

1. Create a Python virtual environment.
2. Install the required dependencies for the CV and pipeline stack.
3. Configure the environment variables and model paths in the repository config files.
4. Run the generation workflow to create baselines.
5. Run inference on sample images to validate the pipeline.

Because repository setup details may vary by environment, the config and pipeline modules should be reviewed before execution.

## Status

This repository is an architecture and pipeline prototype with supporting implementation modules and documentation. It is intended as a practical end-to-end CV workflow for hospitality monitoring and operational auditing.

## License

This project does not currently declare a license in the repository root. If needed, add an appropriate open-source license before public distribution.

