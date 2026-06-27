from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import os
import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from .base_parser import BaseParser
except ImportError:
    from base_parser import BaseParser


class TXTParser(BaseParser):
    """
    Parser for txt files
    """

    type = "txt"

    def __init__(
        self,
        file_path: str = None,
        model=None,
        chunk_size: int = 250,
        chunk_overlap: int = 40,
        separators: List[str] = None,
    ) -> None:
        super().__init__(file_path, model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            "。",
            ".",
            "！",
            "!",
            "？",
            "?",
            "；",
            ";",
            "，",
            ",",
            " ",
            "",
        ]

    def parse(self) -> List[Dict]:
        page_sents = self._to_sentences()

        if not page_sents:
            return None

        self.parse_output = []
        for _, sent in page_sents:
            file_dict = {}
            file_dict["title"] = None
            file_dict["author"] = None
            file_dict["page"] = None
            file_dict["content"] = sent
            file_dict["embedding"] = self.get_embedding(sent)
            file_dict["file_path"] = self.file_path
            file_dict["subject"] = None

            self.parse_output.append(file_dict)

        return self.parse_output

    def _to_sentences(self) -> List[Tuple[int, str]]:
        """
        Parse txt file to text [(pageno, sentence)]
        """
        if not self._check_format():
            self.parse_output = None
            return []

        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # remove hyphens
        raw_text = re.sub(r"-\n(\w+)", r"\1", raw_text)
        raw_text = raw_text.replace("\n", " ")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = [
            chunk.strip() for chunk in splitter.split_text(raw_text) if chunk.strip()
        ]
        return list(map(lambda x: (0, x), chunks))

    @property
    def metadata(self) -> defaultdict:
        # txt files don't have metadata
        if not self._metadata:
            metadata = defaultdict(str)
            self._metadata = metadata

        return self._metadata

    def _check_format(self) -> bool:
        f_path: Path = Path(self.file_path)
        return f_path.exists() and f_path.suffix == ".txt"


if __name__ == "__main__":
    parser = TXTParser(sys.argv[1], None)
    parser._to_sentences()
    # print(parser.parse_output)
    parser.parse()
