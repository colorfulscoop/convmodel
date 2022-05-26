# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add max_len option in `fit` method https://github.com/colorfulscoop/convmodel/pull/21

### Fixed

- Add max_len option in Dataset to restrict the max length of a tensor inputting to a model https://github.com/colorfulscoop/convmodel/pull/20

## [v0.3.0]

### Added

- `output_path` and `save_best_model` parameters to `fit` method. https://github.com/colorfulscoop/convmodel/pull/13
  - `output_path`: when given, model is saved while training in the fit method. Default is None (will not be saved)
  - `save_best_model`: when given with output_path, fit will only save the best model in `output_path`
- `min_new_tokens` option to `generate` method. https://github.com/colorfulscoop/convmodel/pull/16
- eval method in model and cli. https://github.com/colorfulscoop/convmodel/pull/14
- `print_config` option to `fit` cli command. https://github.com/colorfulscoop/convmodel/pull/18

### Changed

- streamlit UI by using side-by-side columns to show conversation. https://github.com/colorfulscoop/convmodel/pull/15
- Change CLI command name from run_streamlit to try. https://github.com/colorfulscoop/convmodel/pull/17

## [v0.2.0]

### Added

- fit method in ConversatonModel. https://github.com/colorfulscoop/convmodel/pull/8
- fit CLI https://github.com/colorfulscoop/convmodel/pull/10

### Changed

- device transfer code in fit method. https://github.com/colorfulscoop/convmodel/pull/9
- document to use MkDocs. https://github.com/colorfulscoop/convmodel/pull/11

### Removed

- Obsoleted modules for bert and gpt2_lm. https://github.com/colorfulscoop/convmodel/pull/7/files

## [v0.1.1] - 2021-08-28

### Added

- build_data_loader method in ConversationDataset class. https://github.com/colorfulscoop/convmodel/pull/5

## [v0.1.0] - 2021-08-28

### Added

- ConversationModel and its usage.
