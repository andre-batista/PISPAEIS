import os
import zipfile
import shutil
import re
from pathlib import Path

def extract_referenced_figures(tex_file_path):
    """
    Extract all figure paths referenced in the LaTeX file
    """
    referenced_figures = set()
    
    with open(tex_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match \includegraphics{path} and \input{path}
    patterns = [
        r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
        r'\\input\{([^}]+)\}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Clean up the path
            fig_path = match.strip()
            
            # Handle different path formats
            if fig_path.startswith('./'):
                fig_path = fig_path[2:]  # Remove ./
            elif fig_path.startswith('../'):
                fig_path = fig_path[3:]  # Remove ../
            
            # Include figures from experiments folder (but not algorithms which are already in paper/)
            if fig_path.startswith('experiments/'):
                referenced_figures.add(fig_path)
    
    return referenced_figures

def create_readme_file():
    """
    Create a README file with compilation instructions
    """
    readme_content = """# Manuscript Compilation Instructions

This ZIP file contains all necessary files to compile the paper "Performance Indicators for Shape and Position Assessment in Electromagnetic Inverse Scattering".

## Files included:
- `paper/main.tex` - Main LaTeX document
- `paper/mybib.bib` - Bibliography file
- `paper/figs/` - Figures used in the paper (local)
- `paper/algorithms/` - Algorithm pseudocode files
- `experiments/` - Figures from experiments referenced in the paper

## Compilation Instructions:

### Option 1: Using pdflatex (recommended)
```bash
cd paper
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

### Option 2: Using latexmk
```bash
cd paper
latexmk -pdf -shell-escape main.tex
```

## Requirements:
- LaTeX distribution with pdflatex
- BibTeX for bibliography processing
- epstopdf package (usually included in LaTeX distributions)
- Shell escape enabled (-shell-escape flag) for automatic EPS to PDF conversion

## Notes:
- The `-shell-escape` flag is required for automatic conversion of EPS figures to PDF
- All figures from the experiments folder are included with their original paths
- The document should compile without any missing figure errors

## File Structure:
The ZIP maintains the original folder structure to ensure all relative paths work correctly:
- `paper/` contains the main document and local figures
- `experiments/` contains figures generated from experimental results
- All paths in the LaTeX file are preserved as in the original project

Generated on: """ + str(Path().absolute()) + """
"""
    return readme_content

def create_manuscript_zip():
    """
    Creates a ZIP file with all necessary files to compile main.tex
    """
    paper_dir = Path("paper")
    experiments_dir = Path("experiments")
    output_zip = "manuscript.zip"
    
    # Check if paper folder exists
    if not paper_dir.exists():
        print("Error: 'paper' folder not found!")
        return
    
    # Check if main.tex exists
    main_tex = paper_dir / "main.tex"
    if not main_tex.exists():
        print("Error: 'main.tex' file not found in paper folder!")
        return
    
    # Extract referenced figures from main.tex
    print("Analyzing main.tex for referenced figures...")
    referenced_figures = extract_referenced_figures(main_tex)
    print(f"Found {len(referenced_figures)} referenced figures from experiments")
    
    # File extensions needed for LaTeX
    tex_extensions = {'.tex', '.bib', '.cls', '.sty', '.bst', '.clo'}
    image_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg'}
    other_extensions = {'.txt', '.md', '.readme'}
    
    all_extensions = tex_extensions | image_extensions | other_extensions
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add README file
        readme_content = create_readme_file()
        zipf.writestr("README.md", readme_content)
        print("Added: README.md")
        
        # Walk through all files in paper folder
        print("\nAdding files from paper folder...")
        for root, dirs, files in os.walk(paper_dir):
            for file in files:
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()
                
                # Include files with relevant extensions
                if file_ext in all_extensions or file == 'main.tex':
                    # Calculate relative path to maintain folder structure
                    arcname = file_path.relative_to(paper_dir.parent)
                    zipf.write(file_path, arcname)
                    print(f"Added: {arcname}")
        
        # Add referenced figures from experiments
        print(f"\nAdding referenced figures from experiments...")
        added_figures = 0
        for fig_path in referenced_figures:
            full_fig_path = Path(fig_path)
            
            if full_fig_path.exists():
                # Add the figure file
                zipf.write(full_fig_path, fig_path)
                print(f"Added figure: {fig_path}")
                added_figures += 1
                
                # Also check for converted PDF version (in case it exists)
                pdf_version = full_fig_path.with_suffix('.pdf')
                if pdf_version.exists():
                    pdf_arcname = str(full_fig_path.with_suffix('.pdf'))
                    zipf.write(pdf_version, pdf_arcname)
                    print(f"Added PDF version: {pdf_arcname}")
            else:
                print(f"Warning: Referenced figure not found: {fig_path}")
        
        print(f"\nSuccessfully added {added_figures} figure files from experiments")
    
    print(f"\nZIP file created: {output_zip}")
    print(f"Location: {Path(output_zip).absolute()}")
    print("\nContents summary:")
    print("- All LaTeX files from paper/ folder")
    print("- All figures from paper/figs/ folder") 
    print(f"- {added_figures} referenced figures from experiments/ folder")
    print("- Algorithms files (zetas.tex, zetap.tex)")

if __name__ == "__main__":
    create_manuscript_zip()