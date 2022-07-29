import os


def create_default_license_file(corpus_path):
    full_path = os.path.join(corpus_path, "LICENSE")
    if not os.path.exists(full_path):
        with open(full_path, "w") as f:
            f.write("")
