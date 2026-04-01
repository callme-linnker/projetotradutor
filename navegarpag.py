def carregar_pagina(doc, x):
    page = doc[x]
    blocos = page.get_text("blocks", sort=True)
    return blocos
