import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crea_campo_105x68(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
    
    # Sfondo bianco (ottimo contrasto con il verde scuro)
    ax.set_facecolor('white')
    
    # Il tuo colore verde bosco scuro
    color_linee = '#1b4332'
    
    # 1. Perimetro esterno (105x68)
    campo = patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor=color_linee, facecolor='none', zorder=5)
    ax.add_patch(campo)

    # 2. Linea di metà campo e cerchio centrale
    ax.plot([52.5, 52.5], [0, 68], color=color_linee, linewidth=2, zorder=5)
    centro_campo = patches.Circle((52.5, 34), 9.15, linewidth=2, edgecolor=color_linee, facecolor='none', zorder=5)
    ax.add_patch(centro_campo)
    ax.scatter(52.5, 34, color=color_linee, zorder=6) # Punto centrale

    # 3. Area di Rigore Sinistra
    area_rigore_sx = patches.Rectangle((0, 34 - 20.16), 16.5, 40.32, linewidth=2, edgecolor=color_linee, facecolor='none', zorder=5)
    area_piccola_sx = patches.Rectangle((0, 34 - 9.16), 5.5, 18.32, linewidth=2, edgecolor=color_linee, facecolor='none', zorder=5)
    ax.add_patch(area_rigore_sx)
    ax.add_patch(area_piccola_sx)
    ax.scatter(11, 34, color=color_linee, zorder=6) # Dischetto rigore sx

    # 4. Area di Rigore Destra
    area_rigore_dx = patches.Rectangle((105 - 16.5, 34 - 20.16), 16.5, 40.32, linewidth=2, edgecolor=color_linee, facecolor='none', zorder=5)
    area_piccola_dx = patches.Rectangle((105 - 5.5, 34 - 9.16), 5.5, 18.32, linewidth=2, edgecolor=color_linee, facecolor='none', zorder=5)
    ax.add_patch(area_rigore_dx)
    ax.add_patch(area_piccola_dx)
    ax.scatter(105 - 11, 34, color=color_linee, zorder=6) # Dischetto rigore dx

    # 5. Archi (Lunette delle aree)
    arco_sx = patches.Arc((11, 34), 18.3, 18.3, theta1=308, theta2=52, linewidth=2, edgecolor=color_linee, zorder=5)
    arco_dx = patches.Arc((105 - 11, 34), 18.3, 18.3, theta1=128, theta2=232, linewidth=2, edgecolor=color_linee, zorder=5)
    ax.add_patch(arco_sx)
    ax.add_patch(arco_dx)

    # --- RIMOZIONE NUMERI E TACCHE ---
    ax.set_xticks([]) 
    ax.set_yticks([]) 

    # Limiti assi
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    
    return ax