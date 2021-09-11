from .cli import CliEntrypoint


if __name__ == "__main__":
    import fire

    fire.Fire(CliEntrypoint)
