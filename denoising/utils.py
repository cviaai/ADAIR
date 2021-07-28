import argparse


def get_root():
    root = "."

    parser = argparse.ArgumentParser(description="root path", add_help=True)
    parser.add_argument("--root", type=str, help="path to the project")
    if parser.parse_args().root is not None:
        root = parser.parse_args().root

    return root
