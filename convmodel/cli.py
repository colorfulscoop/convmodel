import subprocess
from importlib.resources import path
from pkg_resources import resource_filename


class Main:
    def run_streamlit(self, **kwargs):
        options = []
        for key, val in kwargs.items():
            options.extend([f"--{key}", f"{val}"])

        path_to_app = resource_filename("convmodel", "app.py")
        cmd = ["streamlit", "run", path_to_app, *options]
        print(f"Command to execute in subprocess: $", " ".join(cmd))

        subprocess.run(cmd)


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
