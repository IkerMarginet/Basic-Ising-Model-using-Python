# Attention avec H, tsim,  tprint et T


import numpy as np
import matplotlib.pyplot as plt

# Paramètres communs
J, H, Tc = 1, 0.1, 2/np.log(1+np.sqrt(2))
L = 50
tsim, tprint = 100000, 10000
np.random.seed(1756567)


# SIMULATION 1: SANS DOMAINE CIRCULAIRE

print("=== Simulation sans domaine circulaire ===")
T = Tc - 1  # Température T (CHANGER A + OU - 1 SI NECESSAIRE)
Tr = int(1000*T/Tc + 0.5)/1000

# Initialisation rand
S = np.where(np.random.rand(L,L) > 0.5, 1, -1)
M = []

# Simulation
for t in range(1, tsim + 1):
    p, q = np.random.randint(0, L, 2)
    
    # Somme de spins voisins
    somme = (S[(p + 1) % L, q]) + (S[(p - 1) % L, q]) + (S[p, (q + 1) % L]) + (S[p, (q - 1) % L])
    
    Eav = -J * S[p,q] * somme - H * S[p,q]
    DeltaE = -2 * Eav

    if DeltaE < 0 or np.random.rand() < np.exp(-DeltaE / T):
        S[p,q] *= -1

    if t % tprint == 0:
        M.append(S.sum() / (L*L))
        # Affichage dynamique
        plt.figure(1)
        plt.clf()
        plt.spy(S + 1)
        plt.title(f'Configuration aléatoire - T/Tc = {Tr}, t = {t}')
        plt.pause(0.1)

# Affichage final
plt.figure(2, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.spy(S + 1)
plt.title(f'Configuration finale\nT/Tc = {Tr}')

plt.subplot(1, 2, 2)
plt.plot(M)
plt.title('Evolution de l\'aimantation moyenne')
plt.xlabel('t')
plt.ylabel('M/N')

# Calcul de la susceptibilité
ksi = np.sqrt(np.var(M) * L * L)
plt.suptitle(f'Simulation Ising - Configuration aléatoire (Susceptibilité = {ksi:.2f})')
plt.tight_layout()
plt.show()

print("Valeur de ksi =",  ksi)


# SIMULATION 2: AVEC DOMAINE CIRCULAIRE INITIAL

print("\n=== Simulation avec domaine circulaire ===")
T = Tc - 1  # Température réduite
Tr = int(1000*T/Tc + 0.5)/1000

def creer_dom(L, R0):
    """Crée un domaine circulaire de spins -1 dans une mer de spins +1"""
    S = np.ones((L, L))
    centre = L//2
    for i in range(L):
        for j in range(L):
            if (i-centre)**2 + (j-centre)**2 <= R0**2:
                S[i,j] = -1
    return S

def rayon_effectif(S, L):
    """Rayon effectif du domaine circulaire"""
    aire = np.sum(S == -1)
    return np.sqrt(aire/np.pi)

# Initialisation du domaine circulaire
R0 = L//4  # Rayon initial du domaine
S = creer_dom(L, R0)
M, R_ev = [], []

# Simulation
for t in range(1, tsim + 1):
    p, q = np.random.randint(0, L, 2)
    
    # Somme de spins voisins
    somme = (S[(p + 1) % L, q]) + (S[(p - 1) % L, q]) + (S[p, (q + 1) % L]) + (S[p, (q - 1) % L])
    
    Eav = -J * S[p,q] * somme - H * S[p,q]
    DeltaE = -2 * Eav

    if DeltaE < 0 or np.random.rand() < np.exp(-DeltaE / T):
        S[p,q] *= -1

    if t % tprint == 0:
        M.append(S.sum() / (L*L))
        R_ev.append(rayon_effectif(S, L))
        # Affichage dynamique
        plt.figure(3)
        plt.clf()
        plt.spy(S + 1)
        plt.title(f'Domaine circulaire - T/Tc = {Tr}, t = {t}\nRayon = {R_ev[-1]:.1f}')
        plt.pause(0.1)

# Affichage final
plt.figure(4, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.spy(S + 1)
plt.title(f'Configuration finale\nT/Tc = {Tr}')
    
plt.subplot(1, 2, 2)    
plt.plot(R_ev)   
plt.title('Evolution du rayon du domaine')  
plt.xlabel('t')
plt.ylabel('Rayon effectif')    
    
plt.suptitle('Simulation Ising - Domaine circulaire')   
plt.tight_layout()  
plt.show() 