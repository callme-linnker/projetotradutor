from enginetradutor import traduzir_texto
import pymupdf

def reescrever_texto(texto_traduzido, blocos, doc,x):
    page = doc[x]
    blocos = page.get_text("blocks", sort=True)
    for block in blocos:
        x0, y0, x1, y1 = block[:4]  # Pegando as coordenadas do bloco
        rect_do_paragrafo = pymupdf.Rect(x0, y0, x1, y1)
        page.add_redact_annot(rect_do_paragrafo)
        page.apply_redactions()
        page.insert_htmlbox(rect_do_paragrafo, texto_traduzido)
