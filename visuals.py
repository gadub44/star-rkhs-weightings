import seaborn

def set_visual_settings():
    seaborn.set(rc={'axes.axisbelow': False,
                    'axes.edgecolor': 'black',
                    'axes.facecolor': 'None',
                    'axes.grid': False,
                    'axes.labelcolor': 'black',
                    'axes.spines.right': False,
                    'axes.spines.top': False,
                    'figure.facecolor': 'white',
                    'lines.solid_capstyle': 'round',
                    'patch.edgecolor': 'w',
                    'patch.force_edgecolor': True,
                    'text.color': 'black',
                    'xtick.bottom': False,
                    'xtick.color': 'black',
                    'xtick.direction': 'out',
                    'xtick.top': False,
                    'ytick.color': 'black',
                    'ytick.direction': 'out',
                    'ytick.left': False,
                    'ytick.right': False})
    
set_visual_settings()