"""
This script checks for duplicate lines in a given configuration file.

Functions:
    check_for_duplicates(filename):
        Reads the specified file, ignoring empty lines, and identifies any duplicate lines.
        Prints the duplicate lines if found, otherwise indicates no duplicates.

Usage:
    python checkForDuplicates.py <config_file>

Arguments:
    <config_file> : Path to the configuration file to be checked for duplicate lines.
"""
import sys

def check_for_duplicates(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
    seen = set()
    duplicates = set()
    for line in lines:
        if line in seen:
            duplicates.add(line)
        else:
            seen.add(line)
    if duplicates:
        print("Duplicate lines found:")
        for dup in duplicates:
            print(dup)
    else:
        print("No duplicate lines found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config_file>")
        sys.exit(1)
    check_for_duplicates(sys.argv[1])
