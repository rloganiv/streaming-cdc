"""Code for processing medmentions data."""
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import sys
from typing import List, Optional


@dataclass
class Mention:
    start: int
    end: int
    text: str
    semantic_types: List[str]
    entity_id: str


@dataclass
class Document:
    pmid: str
    title: str
    abstract: str
    mentions: List[Mention]
    date: Optional[datetime] = None

    @classmethod
    def from_lines(cls, lines):
        pmid, _, title = lines.popleft().split('|')
        *_, abstract = lines.popleft().split('|')
        mentions = []
        while lines:
            line = lines.popleft()
            _, *fields = line.split('\t')
            mention = Mention(
                start=int(fields[0]),
                end=int(fields[1]),
                text=fields[2],
                semantic_types=fields[3].split(','),
                entity_id=fields[4]
            )
            mentions.append(mention)
        return cls(pmid, title, abstract, mentions)


def parse_pubtator(f):
    """Parses documents from a PubTator-formatted file."""
    instances = []
    lines = deque()
    for line in f:
        if line != '\n':
            lines.append(line.rstrip())
        else:
            yield Document.from_lines(lines)

