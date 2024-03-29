[![Homepage][homepage-shield]][homepage-url]
[![License][license-shield]][license-url]
[![Contributors][contributors-shield]][contributors-url]
[![Code Style][codestyle-shield]][codestyle-url]

# PTE Decode - Python tools for electrophysiology

PTE Decode is an open-source software package for neural decoding.

It builds upon [PTE](https://github.com/richardkoehler/pte) and aims at decoding
intracranial EEG (iEEG) signals such as local field potentials (LFP)
electrocorticography (ECoG).

PTE Decode implements sample-wise decoding and lets you define epochs based on
specific events to avoid circular training.

## Installing PTE Decode

First, get the current development version of PTE using
[git](https://git-scm.com/). Then type the following command into a terminal:

```bash
git clone https://github.com/richardkoehler/pte-decode
```

Use the package manager
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) to set up a
new working environment. To do so, use `cd` in your terminal to navigate to the
PTE root directory and type:

```bash
conda env create -f env.yml
```

This will set up a new conda environment called `pte-decode`.

To activate the environment then type:

```bash
conda activate pte-decode
```

Finally, to install PTE Decode in an editable development version inside your
environment type the following inside the PTE Decode root directory:

```bash
pip install -e .
```

## Usage

```python
import pte_decode

# Examples
```

## Contributing

Please feel free to contribute.

For any minor additions or bugfixes, you may simply create a **pull request**.

For any major changes, make sure to open an **issue** first. When you then
create a pull request, be sure to **link the pull request** to the open issue in
order to close the issue automatically after merging.

To contribute, consider installing the full conda development environment to
include such tools as black, pylint and isort:

```bash
conda env create -f env_dev.yml
conda activate pte-decode-dev
```

Continuous Integration (CI) including automated testing are set up.

## License

PTE is licensed under the [MIT license](license-url).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]:
  https://img.shields.io/github/contributors/richardkoehler/pte.svg?style=for-the-badge
[contributors-url]: https://github.com/richardkoehler/pte/graphs/contributors
[license-shield]:
  https://img.shields.io/static/v1?label=License&message=MIT&logoColor=black&labelColor=grey&logoWidth=20&color=yellow&style=for-the-badge
[license-url]: https://github.com/richardkoehler/pte/blob/main/LICENSE/
[codestyle-shield]:
  https://img.shields.io/static/v1?label=CodeStyle&message=black&logoColor=black&labelColor=grey&logoWidth=20&color=black&style=for-the-badge
[codestyle-url]: https://github.com/psf/black
