# Makefile 内容
SRC=overview.md
TEX=input.tex
PDF=MF_document.pdf

all:
	pandoc $(SRC) \
	--metadata-file=meta.yaml \
	-o $(TEX)

# all:
# 	pandoc $(SRC) \
# 	--template=template.tex \
# 	--pdf-engine=xelatex \
# 	--metadata-file=meta.yaml \
# 	-o $(PDF)
