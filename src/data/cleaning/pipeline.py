from src.data.cleaning import run_un_gutenberg as un_gutenberg
from src.data.cleaning import run_data_cleanup as clean

pipeline_steps = [
    un_gutenberg,
    clean
]


def main():
    for step in pipeline_steps:
        step()


if __name__ == "__main__":
    main()
