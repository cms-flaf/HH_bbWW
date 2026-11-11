#!/usr/bin/env python3
"""The datacard cards' `include:` lists agree, so no uncertainty file is forgotten in one card.

The cards list the files they include explicitly (StatInference's includes do not nest), one per
uncertainty topic: config/Datacards/systematics/ (shared by DL, SL and the CI cards) and
config/Datacards/DL/systematics/ (DL only). A file left out of one card's list is no error for
StatInference -- the card just fits without those uncertainties -- so this checks:

- every included file exists;
- the cards of one kind (the DL spin cards, the SL spin cards) include the same topics;
- every card includes every topic of systematics/, and every DL card every topic of
  DL/systematics/.

A topic is a file name without a version suffix (`_v<digits>...`), so a card may point to another
version of a topic file. Only PyYAML is needed.

    python3 test/test_datacard_includes.py
"""

import glob
import os
import re
import unittest

import yaml

ANA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATACARDS = os.path.join(ANA, "config", "Datacards")

#: the cards of one kind, which must include the same topics
KINDS = {
    "DL": ["DL/dl_spin0.yaml", "DL/dl_spin2.yaml"],
    "SL": ["SL/sl_spin0.yaml", "SL/sl_spin2.yaml"],
    "CI": ["CI/CI_card_DL.yaml", "CI/CI_card_SL.yaml"],
}
#: topic folders, relative to config/Datacards, and the kinds of card that include all of them
TOPIC_DIRS = {"systematics": ["DL", "SL", "CI"], "DL/systematics": ["DL"]}


def topic(path):
    """(folder relative to config/Datacards, file stem without a version suffix)."""
    folder = os.path.relpath(os.path.dirname(path), DATACARDS)
    stem = os.path.splitext(os.path.basename(path))[0]
    return folder, re.sub(r"_v\d\w*$", "", stem)


def includes(card):
    """The absolute paths a card includes (relative to the card, as StatInference reads them)."""
    path = os.path.join(DATACARDS, card)
    with open(path) as f:
        listed = yaml.safe_load(f).get("include") or []
    if isinstance(listed, str):
        listed = [listed]
    return [os.path.normpath(os.path.join(os.path.dirname(path), i)) for i in listed]


class Includes(unittest.TestCase):
    def test_every_included_file_exists(self):
        for cards in KINDS.values():
            for card in cards:
                for path in includes(card):
                    with self.subTest(card=card, include=path):
                        self.assertTrue(os.path.isfile(path), f"{card} includes {path}")

    def test_cards_of_one_kind_include_the_same_topics(self):
        for kind, cards in KINDS.items():
            topics = {card: sorted(topic(p) for p in includes(card)) for card in cards}
            with self.subTest(kind=kind):
                first = topics[cards[0]]
                for card in cards[1:]:
                    self.assertEqual(topics[card], first, f"{card} vs {cards[0]}")

    def test_every_card_includes_every_topic_of_its_folders(self):
        for folder, kinds in TOPIC_DIRS.items():
            files = glob.glob(os.path.join(DATACARDS, folder, "*.yaml"))
            self.assertTrue(files, f"no topic files in {folder}")
            wanted = {topic(f) for f in files}
            for kind in kinds:
                for card in KINDS[kind]:
                    with self.subTest(folder=folder, card=card):
                        missing = wanted - {topic(p) for p in includes(card)}
                        self.assertFalse(
                            missing,
                            f"{card} does not include {sorted(t for _, t in missing)} of "
                            f"{folder}/",
                        )


if __name__ == "__main__":
    unittest.main()
