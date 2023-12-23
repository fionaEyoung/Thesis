#!/usr/bin/python3

# Adapted from https://gitlab.com/-/snippets/1880361 by Marc Schreiber (https://gitlab.com/schrieveslaach)
import os
from panflute import *
import re

acronyms = {}

refcounts = {}

def resolveAcronyms(elem, doc):
    if isinstance(elem, Span) and "acronym-label" in elem.attributes:
        label = elem.attributes["acronym-label"]

        if label in acronyms:
            # this is the case: "singular" in form and "long" in form:
            value = acronyms[label][1]
            short = acronyms[label][0]

            form = elem.attributes["acronym-form"]
            if label in refcounts and "short" in form:
                if "singular" in form:
                    value = short
                else:
                    value = short + "s"

            elif "full" in form or "short" in form:
                # remember that label has been used
                if "short" in form:
                    refcounts[label] = True

                if "singular" in form:
                    value = value + " (" + short + ")"
                else:
                    value = value + "s (" + short + "s)"

            elif "abbrv" in form:
                if "singular" in form:
                    value = short
                else:
                    value = short + "s"

            return Span(Str(value))

def loadAcronyms():
    pattern = re.compile(r"\\newacronym(\[.*\])?\{(?P<label>[A-Za-z]+)\}\{(?P<short>[A-Za-z]+)\}\{(?P<value>[A-Za-z 0-9\-]+)\}")

    d = os.path.dirname(__file__)
    filename = os.path.join(d, 'setup/AcronymsAndGlossary.tex')
    with open(filename, 'r', encoding='utf-8') as acronymsFile:
        for line in acronymsFile:
            match = pattern.match(line)
            if match:
                acronyms[match.group('label')] = (match.group('short'), match.group('value'))

def main(doc=None):
    loadAcronyms()
    return run_filter(resolveAcronyms, doc=doc)


if __name__ == "__main__":
    main()
