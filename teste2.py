import pymupdf
def reescrever_pag(batch, doc):
    for traducao, coords, pag_num in batch:
        x0, y0, x1, y1 = coords
        rect = pymupdf.Rect(x0, y0, x1, y1)
        page = doc[pag_num]

        page.add_redact_annot(rect)
        page.apply_redactions()

        texto_html = f"""
        <p style="font-family: sans-serif; font-size: 9pt; text-align: justify;">
            {traducao}
        </p>
        """
        page.insert_htmlbox(rect, texto_html)

    doc.save("documento_traduzido.pdf")
    paginas = {pag for _, _, pag in batch}
    print(f"Documento salvo — {len(paginas)} páginas processadas.")
