"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("dmff").rglob("*.py")):  # 

    if path.name.startswith("_"):
        continue
    
    module_path = path.relative_to('dmff').with_suffix("")  # 

    doc_path = path.relative_to('dmff').with_suffix(".md")  # 

    full_doc_path = Path("refs", doc_path)  # 

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  # 
        continue
    elif parts[-1] == "__main__":
        continue
    
    nav[parts] = doc_path.as_posix()
    print(full_doc_path)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  # 

        identifier = ".".join(parts)  # 

        print("::: dmff." + identifier, file=fd)  # 


    mkdocs_gen_files.set_edit_path(full_doc_path, path)  # 

with mkdocs_gen_files.open("refs/SUMMARY.md", "w") as nav_file:  # 

    nav_file.writelines(nav.build_literate_nav())  # 

