from stlitepack import pack, setup_github_pages
from stlitepack.pack import list_files_in_folders

files_to_link = list_files_in_folders(["examples", "resources"], recursive=True)

print(files_to_link)

pack(
    "Introduction.py",
    extra_files_to_link=files_to_link,
    prepend_github_path="Bergam0t/simpy_visualisation",
    run_preview_server=True,
    requirements=[
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "simpy==4.0.2,<5",
        "vidigi==1.0.0",
        "arrow",
        "setuptools",
    ],
)

setup_github_pages()
