#!/usr/bin/env bash

pandoc report.md --bibliography=references.bib --citeproc -o report.pdf
