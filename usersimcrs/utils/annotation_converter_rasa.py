"""Generates Rasa NLU files from DialogueKit annotated dialogues JSON format."""

from dialoguekit.utils.annotation_converter_dialoguekit_to_rasa import (
    AnnotationConverterRasa,
)
import argparse
import sys
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-source", type=str, help="Path to the annotated dialogues file."
    )
    parser.add_argument(
        "-destination", type=str, help="Path to the destination folder."
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        sys.exit("FileNotFound: {file}".format(file=args.source))
    if not os.path.exists(args.destination):
        sys.exit("FileNotFound: {file}".format(file=args.destination))

    converter = AnnotationConverterRasa(args.source, args.destination)
    converter.read_original()
    converter.run()
