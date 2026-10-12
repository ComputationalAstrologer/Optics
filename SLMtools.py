import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class SLMSetup:
    """Configuration du modulateur spatial de lumière (SLM)."""
    height_width: Tuple[int, int] = (1536, 2048)
    scale_options: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0), (0.0, 255.0)])
    active_scale: Tuple[float, float] = (0.0, 255.0) # S'adapte automatiquement (ex: 0-1 ou 0-255)
    screen_offset_x: int = 1920
    screen_offset_y: int = 0
    dpi: int = 100

    # Constantes de calibration empiriques du banc
    c_calib_x: float = 0.5
    c_calib_y: float = 0.5

SLM_SETUP = SLMSetup()
_fig, _ax, _im = None, None, None

def initialiser_slm(setup: SLMSetup = SLM_SETUP):
    """Initialise l'affichage plein écran sur le SLM."""
    global _fig, _ax, _im
    mpl.rcParams['keymap.fullscreen'] = ''
    mpl.rcParams['toolbar'] = 'None'

    height, width = setup.height_width
    vmin, vmax = setup.active_scale

    _fig, _ax = plt.subplots(figsize=(width / setup.dpi, height / setup.dpi), dpi=setup.dpi)
    _fig.canvas.manager.set_window_title('SLM - Ajustement Amplitude & Phase')
    _fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _ax.axis('off')

    _im = _ax.imshow(np.zeros((height, width)), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

    mgr = plt.get_current_fig_manager()
    try:
        mgr.window.wm_geometry(f"+{setup.screen_offset_x}+{setup.screen_offset_y}")
        mgr.full_screen_toggle()
    except AttributeError:
        pass
    plt.ion()
    plt.show()
    plt.pause(0.1)

def envoyer_au_slm(tableau: np.ndarray, setup: SLMSetup = SLM_SETUP):
    """Envoie la matrice au SLM."""
    global _fig, _im
    if tableau.shape != setup.height_width:
        raise ValueError(f"Taille attendue : {setup.height_width}, reçu : {tableau.shape}")
    if _fig is None or not plt.fignum_exists(_fig.number):
        initialiser_slm(setup)
    _im.set_data(tableau)
    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()

def generer_masque_frequence(fx: float, fy: float, amp: float = 0.5, phi: float = 0.0, setup: SLMSetup = SLM_SETUP) -> np.ndarray:
    """
    Génère un profil d'amplitude sinusoïdal continu prenant en compte active_scale.

    Parameters:
    -----------
    fx, fy : float
        Fréquences spatiales (nombre de périodes sur la largeur/hauteur du SLM).
    amp : float
        Amplitude scalaire de l'oscillation. Maximum de 0.5 (représente 100% de la dynamique de active_scale).
    phi : float
        Phase spatiale en radians (ex: phi = np.pi/2 transforme le profil en cosinus).
    """
    if amp > 0.5:
        raise ValueError("L'amplitude scalaire 'amp' ne peut pas dépasser 0.5 pour éviter la saturation du SLM.")

    h, w = setup.height_width
    vmin, vmax = setup.active_scale

    # Coordonnées des pixels
    y, x = np.indices((h, w))

    # Calcul des paramètres physiques basés sur active_scale
    milieu = (vmax + vmin) / 2.0
    dynamique_totale = vmax - vmin

    # Génération du profil physique directement dans la bonne échelle (active_scale)
    profil = milieu + (amp * dynamique_totale) * np.sin(2 * np.pi * (fx * x / w + fy * y / h) + phi)

    return profil

def generer_mire_test(type_mire: str, setup: SLMSetup = SLM_SETUP) -> np.ndarray:
    """Génère des mires géométriques pour valider l'alignement sur le banc optique en respectant active_scale."""
    h, w = setup.height_width
    vmin, vmax = setup.active_scale

    # On initialise l'image avec la valeur minimale autorisée (et non 0)
    img = np.full((h, w), vmin, dtype=float)

    if type_mire == 'cross':
        # Une croix centrée de 1 pixel de large + bordure de 1 pixel tout autour du SLM
        img[h//2, :] = vmax
        img[:, w//2] = vmax
        img[0, :] = vmax; img[-1, :] = vmax; img[:, 0] = vmax; img[:, -1] = vmax
        print("Mire générée : Croix centrale et bordure externe (Vérification des limites du SLM).")

    elif type_mire == 'checkerboard':
        # Échiquier de carrés de 64x64 pixels (Idéal pour détecter les distorsions géométriques)
        taille_case = 64
        y, x = np.indices((h, w))
        # Sélectionne les cases blanches (1) ou noires (0)
        masque_cases = ((y // taille_case) + (x // taille_case)) % 2
        # Assigne vmax aux cases blanches, conserve vmin sur les autres
        img[masque_cases == 1] = vmax
        print("Mire générée : Échiquier 64x64 pixels (Vérification du rapport d'aspect).")

    elif type_mire == 'grating':
        # Réseau binaire de pas de 2 pixels (Alternance stricte ligne par ligne)
        # Idéal pour valider le mapping pixel-à-pixel (Mesure du max d'angle de diffraction)
        y, x = np.indices((h, w))
        masque_lignes = x % 2
        img[masque_lignes == 1] = vmax
        print("Mire générée : Réseau binaire de pas 2 pixels (Validation absolue du mapping 1:1).")

    else:
        raise ValueError("Type de mire inconnu. Choisissez parmi: 'cross', 'checkerboard', 'grating'")

    return img

def calculer_masque_depuis_position_focale(dx_cam: float, dy_cam: float, amp: float = 0.5, phi: float = 0.0, setup: SLMSetup = SLM_SETUP) -> np.ndarray:
    """Calcule le masque à partir du décalage caméra, de l'amplitude et de la phase spatiale."""
    fx = dx_cam * setup.c_calib_x
    fy = dy_cam * setup.c_calib_y
    return generer_masque_frequence(fx, fy, amp=amp, phi=phi, setup=setup)

# =============================================================================
# BLOC DE TEST SUR LE BANC OPTIQUE
# =============================================================================
if __name__ == "__main__":
    import time

    SLM_SETUP.screen_offset_x = 1920

    # Changement d'échelle à la volée pour valider la prise en compte de active_scale
    SLM_SETUP.active_scale = (0.0, 1.0)  # On passe en mode normalisé [0.0 , 1.0]
    print(f"Échelle active configurée : {SLM_SETUP.active_scale}")

    decalage_x_pixels = 50.0
    decalage_y_pixels = 20.0

    # Génération du masque (avec amp=0.5, l'oscillation ira précisément de 0.0 à 1.0)
    masque = calculer_masque_depuis_position_focale(decalage_x_pixels, decalage_y_pixels, amp=0.5, phi=0.0)

    print(f"Vérification des valeurs de la matrice générée : Min = {masque.min():.2f}, Max = {masque.max():.2f}")

    print("Envoi au SLM...")
    envoyer_au_slm(masque)

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
