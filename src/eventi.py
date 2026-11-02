import numpy as np

def genera_eventi(n_totale=None, manuali=None):
    """
    n_totale: numero totale di eventi per la partita.
    manuali: lista di tuple [(x1, y1), (x2, y2)] inserite a mano.
    """
    eventi = []
    if manuali:
        eventi.extend(manuali)
    
    # Calcoliamo quanti ne mancano per arrivare a n_totale
    rimanenti = max(0, n_totale - len(eventi))
    
    # --- MODIFICA PER COPERTURA TOTALE ---
    # Media: centro del campo (52.5, 34)
    # Deviazione X: 35.0 (permette di arrivare facilmente a 0 e 105)
    # Deviazione Y: 22.0 (permette di coprire bene le fasce laterali)
    x_random = np.random.uniform(0, 105.0, rimanenti)
    y_random = np.random.uniform(0, 68, rimanenti)

    # .clip(min, max) è fondamentale ora che abbiamo allargato la distribuzione,
    # perché molti punti teorici cadrebbero fuori dal campo.
    # Clip li "schiaccia" esattamente sulle linee di fondo e laterali.
    x_random = np.clip(x_random, 0, 105)
    y_random = np.clip(y_random, 0, 68)
    
    eventi.extend(list(zip(x_random, y_random)))
    
    return np.array(eventi, dtype=np.float32)