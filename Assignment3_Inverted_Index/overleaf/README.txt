OVERLEAF INSTRUCTIONS

1. Upload all files from this folder into one Overleaf project.
2. The main document is main.tex.
3. Open Menu -> Settings -> Compiler and select XeLaTeX.
4. Click Recompile.

The included Amiri font files make Arabic rendering independent of Overleaf's
installed font list. The latexmkrc also requests XeLaTeX automatically. If Overleaf still
shows the fontspec/pdfLaTeX error, changing the compiler in Menu -> Settings is
the definitive fix.

Do not select pdfLaTeX. The report uses fontspec and polyglossia so Arabic is
connected and right-to-left.
