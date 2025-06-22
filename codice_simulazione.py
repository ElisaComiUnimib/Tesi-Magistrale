# %% [markdown]
# Codice per la simulazione della rete stradale del quartiere Bicocca, in particolare dell'area che circonda gli edifici U5, U9 e Piazza della scienza.

# %%
# Importiamo i pacchetti necessari
import numpy as np
import matplotlib.pyplot as plt
from pickle import *

# %% [markdown]
# Iniziamo con il definire le funzioni che utilizzeremo in seguito.

# %%
# Definizione del flusso lungo la rete stradale. E' uguale su tutte le vie
# poiché per come è stato scelto dipende solo dal limite di velocità.
def flux(u:np.array) -> np.array:
    global V_max
    return np.array(V_max*u*(1-u))

V_max = 125/9 # Velocità massima di 50 km/h, in metri al secondo 125/9 m/s (o 30 km/h sono 25/3 m/s )
sigma = 0.5 # Punto di massimo per il flusso

# %% [markdown]
# Definiamo alcune funzioni che utilizzeremo in seguito per i dati al bordo.

# %%
def E_sin(t: np.array) -> np.array:
  return np.square(np.sin((np.pi/120)*t))*0.4+0.5

def E_cos(t: np.array) -> np.array:
  return np.square(np.cos((np.pi/600)*t))*0.2+0.1

def E_exp(t: np.array) -> np.array:
  return np.exp(-t/1800)*0.7+0.1

# %%
def max_f(density, incoming=True) -> float: # Definizione dei gamma_max per i flussi all'incrocio
    global flux, sigma
    if (density < sigma and incoming) or (density > sigma and not incoming):
        return flux(density)
    else:
        return flux(sigma)

# %% [markdown]
# Codice di risoluzione dell'incrocio 1x2

# %%
def solution_1x2(densities:list, alfa: float) -> np.array:
  global max_f
  f_bar = np.zeros(3) # Flussi all'incrocio
  gamma_max = [max_f(densities[i], i == 0) for i in range(3)]
  f_bar[0] = min(gamma_max[0], gamma_max[1]/alfa, gamma_max[2]/(1-alfa))
  f_bar[1] = alfa*f_bar[0]
  f_bar[2] = (1-alfa)*f_bar[0]
  return f_bar

# %% [markdown]
# Codice di risoluzione dell'incrocio 2x1

# %%
def solution_2x1(densities:list, q:float) -> np.array:
    global  max_f, is_acceptable
    f_bar = np.zeros(3)
    gamma_max = [max_f(densities[i], i<2) for i in range(3)]
    if gamma_max[0] + gamma_max[1] <= gamma_max[2]: # Non è necessaria la precedenza
      f_bar[0] = gamma_max[0]
      f_bar[1] = gamma_max[1]
      f_bar[2] = gamma_max[0]+ gamma_max[1]
    else: # Serve la precedenza
      point = [gamma_max[2]*q/(1+q), gamma_max[2]/(1+q), gamma_max[2]]
      if point[0] <= gamma_max[0] and point[1] <= gamma_max[1]:
        f_bar = np.array(point)
      else:
        p_int = [[gamma_max[0], gamma_max[2]- gamma_max[0]], [gamma_max[2]- gamma_max[1], gamma_max[1]]]
        if p_int[0][1] > gamma_max[1] or p_int[1][0] > gamma_max[0]:
          f_bar[0] = gamma_max[0]
          f_bar[1] = gamma_max[1]
          f_bar[2] = gamma_max[2]
        else:
          dist1 = np.linalg.norm([point[0]- p_int[0][0], point[1] - p_int[0][1]])
          dist2 = np.linalg.norm([point[0]- p_int[1][0], point[1]- p_int[1][1]])
          if dist1 <= dist2:
            f_bar = np.array([p_int[0][0], p_int[0][1], gamma_max[2]])
          else:
            f_bar = np.array([p_int[1][0], p_int[1][1], gamma_max[2]])
    return f_bar

# %% [markdown]
# Codice dell'incrocio 2x2

# %%
def is_acceptable(point:tuple, gamma_max:list, alpha: float, beta:float) -> bool:

    positivity = point[0] >= 0. and point[1] >= 0.
    constraint_12 = point[0] <= gamma_max[0] and point[1] <= gamma_max[1]
    contraint_3 = alpha*point[0] + beta*point[1] <= gamma_max[2]
    contraint_4 = (1-alpha)*point[0] + (1-beta)*point[1] <= gamma_max[3]

    return positivity and constraint_12 and contraint_3 and contraint_4

def solution_2x2(densities:list, par: list) -> np.array:
    global max_f, is_acceptable
    alpha = par[0]
    beta = par[1]
    gamma_max = [max_f(densities[i], i<2) for i in range(4)]

    points_omega = []
    # asse verticale
    points_omega.append((0., min(gamma_max[1], gamma_max[2]/beta, gamma_max[3]/(1-beta))))

    # asse orizzontale
    points_omega.append((min(gamma_max[0], gamma_max[2]/alpha, gamma_max[3]/(1-alpha)), 0.))

    # intersezione I_1 e I_3
    point = (gamma_max[0], gamma_max[2]/beta - alpha*gamma_max[0]/beta)
    if is_acceptable(point, gamma_max, alpha, beta):
        points_omega.append(point)

    # intersezione I_1 e I_4
    point = (gamma_max[0], gamma_max[3]/(1-beta) - (1-alpha)*gamma_max[0]/(1-beta))
    if is_acceptable(point, gamma_max, alpha, beta):
        points_omega.append(point)

    # intersezione I_2 e I_3
    point = (gamma_max[2]/alpha - beta*gamma_max[1]/alpha, gamma_max[1])
    if is_acceptable(point, gamma_max, alpha, beta):
        points_omega.append(point)

    # intersezione I_2 e I_4
    point = (gamma_max[3]/(1-alpha) - (1-beta)*gamma_max[1]/(1-alpha), gamma_max[1])
    if is_acceptable(point, gamma_max, alpha, beta):
        points_omega.append(point)

    # intersezione I_3 e I_4
    point = ((beta-1)*gamma_max[2]/(beta-alpha) + beta*gamma_max[3]/(beta-alpha),
             (1-alpha)*gamma_max[2]/(beta-alpha) - alpha*gamma_max[3]*(beta-alpha))
    if is_acceptable(point, gamma_max, alpha, beta):
        points_omega.append(point)

    sum_fluxes = [a+b for (a, b) in points_omega]
    m_flux = max(sum_fluxes)
    index = sum_fluxes.index(m_flux)

    bar_f = list(points_omega[index])
    bar_f.append(alpha*bar_f[0] + beta*bar_f[1])
    bar_f.append((1-alpha)*bar_f[0] + (1-beta)*bar_f[1])

    bar_f = np.array(bar_f)

    return bar_f

# %% [markdown]
# Scriviamo u  algoritmo che con le informazioni su un incrocio, calcola i flussi delle strade coinvolte. Passiamo le informazioni in questo modo:
# $$[\hbox{tipo di incrocio}, \hbox{strade entranti}, \hbox{strade uscenti}, \hbox{parametri}]$$
# dove:
# 1. Tipo di incrocio può assumere tre valori: 12 (1x2), 21 (2x1), 22 (2x2);
# 2. Strade entranti: [numero strada, numero strada]. Nel caso in cui l'incrocio coinvolga una sola strada entrante si ha solo una entrata.
# 3. Strade uscenti: [numero strada, numero strada]. Nel caso in cui l'incrocio coinvolga una sola strada uscente si ha solo una entrata.
# Le strade sono salvate in un vettore in tale ordine: strade di tipo $I$ da 1 a 32, strade di tipo $j$ da 1 a 10, strade di tipo $r$ da 1 a 8, strade di tipo $E$ da 1 a 11. In totale sono 61 strade e la loro numerazione segue l'ordine da 0 a 60 che corrisponde all'indice nel vettore in cui sono salvate le informazioni.
# 4. Parametri: [par_1] se incrocio 1x2 o 2x1, [par_1,par_2] se incrocio 2x2.

# %%
def soluzione_incroci(info: list, roads:list, f_entrata: np.array, f_uscita : np.array) -> list:
    # f_entrata e f_uscita sono dei vettori che ad ogni strada salvano gli eventuali valori del flusso agli incroci se sono coinvolte come strade entranti (f_entrata) o
    # come strade uscenti (f_uscita). Tali valori sono salvati anche in un vettore f_bar che ad ogni incrocio salva i flussi delle strade coinvolte.
    assert info[0] in [12,21,22]
    if info[0] == 12: # Incrocio 1x2
        densities = [roads[info[1][0]][-1],roads[info[2][0]][0],roads[info[2][1]][0]]
        [f_entrata[info[1][0]], f_uscita[info[2][0]], f_uscita[info[2][1]]] = solution_1x2(densities, info[3][0])
    elif info[0] == 21: # Incrocio 2x1
        densities = [roads[info[1][0]][-1],roads[info[1][1]][-1],roads[info[2][0]][0]]
        [f_entrata[info[1][0]], f_entrata[info[1][1]], f_uscita[info[2][0]]]= solution_2x1(densities, info[3][0])
    else: #Incrocio 2x2
        densities = [roads[info[1][0]][-1],roads[info[1][1]][-1], roads[info[2][0]][0], roads[info[2][1]][0]]
        [f_entrata[info[1][0]], f_entrata[info[1][1]], f_uscita[info[2][0]], f_uscita[info[2][1]]] = solution_2x2(densities, info [3])
        return [f_entrata , f_uscita]


# %% [markdown]
# Creiamo una funzione per definire i semafori.

# %%
def semafori(t:float, n:int) -> bool:
    assert t>= 0 and n in [1,2]
    if n == 1:
        return t%45 <= 25
    else:
        return t%75 <= 40

# %% [markdown]
# Schema numerico per le strade: questa applica il metodo numerico a volume finito per ciascuna strada.

# %%
def numerical_scheme_on_roads(u:np.array, f:callable, lam:float, i:int, colore_S1:bool, colore_S2:bool,
                              flux_left = None, flux_right = None, bordo = 0) -> np.array:
    u_plus = np.concatenate((u[1:], np.array([u[-1]])))
    u_minus = np.concatenate((np.array([u[0]]), u[:-1]))

    if colore_S1: # S1 è verde, perciò le strade coinvolte fluiscono senza ostacoli
      if i in[17,19]:
        u_minus = np.concatenate((np.array([bordo]), u[:-1]))
      if i in [16,20]:
        u_plus = np.concatenate((u[1:], np.array([bordo])))
    else: # S1 è rosso quindi il flusso all'incrocio è nullo
      if i in [17,19]:
          flux_left = 0
      if i in [16,20]:
          flux_right = 0

    if colore_S2: # La rete include l'incrocio S15"'
       if i in [22,25]:
          flux_right = 0

    res = u - lam * (f(u, u_plus, lam) - f(u_minus, u, lam))

    # Ridefiniamo eventualmente i valori nelle due celle estreme usando flux_left e flux_right se definiti maggiori di zero (infatti sono inizializzati uguali a -1, nel caso
    # in cui la strada entra o esce da un incrocio questo valore sarà modificato)
    if flux_right >= 0:
        res[-1] = u[-1] - lam * (flux_right  - f(u[-2], u[-1], lam))
    if flux_left >= 0:
        res[0] = u[0] - lam * (f(u[0], u[1], lam) - flux_left)
    return res

# %% [markdown]
# Costruiamo ora l'algoritmo che risolve l'intera rete stradale. Innanzitutto calcola i flussi agli incroci e poi rinnova le informazioni sulle strade.

# %%
def scheme(roads:list, f:callable, lam:float, colore_S1: bool, colore_S2: bool) -> list:
    global incroci, n_strade_tot_rete
    f_entrata = np.ones(61)*(-1)
    f_uscita = np.ones(61)*(-1)
    # Calcoliamo i flussi agli incroci
    if colore_S2:
        for i in range(len(incroci)):
            if i not in [20,21]:
                [f_entrata, f_uscita] = soluzione_incroci(incroci[i],roads, f_entrata, f_uscita)
    else:
        for i in range(len(incroci)):
            if i not in [22]:
                [f_entrata, f_uscita] = soluzione_incroci(incroci[i],roads, f_entrata, f_uscita)

    # Aggiorniamo la soluzione sulla rete stradale
    res = []
    bordoS1 = [roads[19][0], roads[20][-1], 0, roads[16][-1], roads[17][0]]
    for i in range(n_strade_tot_rete): # Non aggiorniamo le strade di tipo E
        if i in [16,17,19,20]:
            res.append(numerical_scheme_on_roads(roads[i], f, lam, i, colore_S1, colore_S2, flux_left=f_uscita[i], flux_right=f_entrata[i], bordo = bordoS1[i-16]))
        else:
            res.append(numerical_scheme_on_roads(roads[i], f, lam, i, colore_S1, colore_S2, flux_left=f_uscita[i], flux_right=f_entrata[i]))
    return res

# %% [markdown]
# Riportiamo due diversi flussi numerici che possiamo utilizzare: Lax-Friedrichs e Godunov.

# %%
def Lax_Friedrichs(density1, density2,lam:float):
    res = 0.5 * (flux(density1) + flux(density2)) - 0.5 * (density2 - density1)/lam
    return res

# %%
def Godunov(u_left:float, u_right:float, lam: float) -> float:
    if u_left == u_right:
        return flux(u_left)
    elif u_left > u_right: # Rarefazione (il flusso è concavo)
        if u_left <= sigma:
            return flux(u_left)
        elif u_right >= sigma:
            return flux(u_right)
        else:
            return flux(sigma)
    else: # u_left < u_right: shock (il flusso è concavo)
        # velocità dello shock (Rankine-Hugoniot)
        lam = (flux(u_left) - flux(u_right)) / (u_left - u_right)
        if lam <= 0:
            return flux(u_right)
        else:
            return flux(u_left)

# Creiamo la funzione che lavora con gli array di Numpy
vectGodunov = np.vectorize(Godunov)

# %% [markdown]
# Iniziamo a inserire i dati, ovvero la lunghezza delle strade della rete.

# %%
I = np.zeros(32) # Strade del tipo I: strade principali interne alla rete
for i in range(len(I)+1):
    if i in [1,2,12,13,14,15,27,28,29,30]:
        I[i-1] = 170
    elif i in [5,6]:
        I[i-1] = 450
    elif i in [9,11,16,19,22]:
        I[i-1] = 150
    elif i in [17,18,20,21,23,24,25,26]:
        I[i-1] = 85
    elif i in [10,31,32]:
        I[i-1] = 46
I[2] = 140
I[3] = 110
I[6] = 56
I[7] = 16

j = np.ones(10)*8 # Strade del tipo j: strade interne agli incroci

r = np.zeros(8) # Strade del tipo r: strade interne alle rotonde
for i  in range(len(r)+1):
    if i in [7,8]:
        r[i-1] = 4
    elif i in [3,6]:
        r[i-1] = 12
r[0] = 8
r[1] = 10
r[3] = 17
r[4] = 14

# Le strade di tipo E sono strade esterne e non ci interessa la loro lunghezza

# %% [markdown]
# Scegliamo gli step spaziali per ciascuna strada.

# %%
dx =[I[i]/100 for i in range(len(I))] # Per le strade interne prendiamo uno step del 1%
dx.extend(j[k]/10 for k in range(len(j))) # Per le strade degli incroci prendiamo un step del 10%
dx.extend(r[k]/10 for k in range(len(r))) # Per le strade delle rotonde prendiamo uno step del 10%
dx_min = min(dx)

xx = [np.arange(dx[i]/2, I[i], dx[i]) for i in range(len(I))]
xx.extend(np.arange(dx[i+len(I)]/2, j[i], dx[i+len(I)]) for i in range(len(j)))
xx.extend(np.arange(dx[i+len(I)+len(j)]/2, r[i], dx[i+len(I)+len(j)]) for i in range(len(r)))

# %% [markdown]
# Per scegliere lo step temporale utilizziamo la condizione CFL.

# %%
max_df_global = V_max

dt_lim = dx_min/max_df_global
dt = dt_lim - 0.0001 # Soddisfa la CFL

# %% [markdown]
# Abbiamo costruito un vettore con tutte le informazioni sulle strade, nel seguente ordine: prima tutte le I da 1 a 32, poi tutte le j da 1 a 10, poi tutte le r da 1 a 8 e infine tutte le esterne da 1 a 11. Allora anche per gli incroci il numero della strada entrante o uscente rispetterà l'indice di questa lista python (parte da 0 fino a 60).

# %%
# Inseriamo i dati degli incroci
par = [] # Lista dei parametri per ciascun incrocio
par.append([1/3]) # J1
par.append([0.3,0.9]) # J2'
par.append([0.8]) # J2"
par.append([1/3]) # J3
par.append([0.5]) # J4
par.append([1/3]) # J5
par.append([0.5]) # J6
par.append([1/3]) # J7
par.append([0.5]) # J8
par.append([3]) # J9
par.append([0.8,0.1]) # J10'
par.append([0.8]) # J10"
par.append([0.9,0.5]) # J11'
par.append([3]) # J11"
par.append([0.8,0.3]) # J12'
par.append([0.5,0.8]) # J12"
par.append([0.4,0.7]) # J13'
par.append([0.4,0.5]) # J13"
par.append([0.3]) # J14'
par.append([0.4,0.9]) # J14"
par.append([0.5,0.1]) # J15'
par.append([0.5]) # J15"
par.append([0.5]) # J15"'
par.append([3]) # J16'
par.append([0.6,0.9]) # J16"
par.append([0.8]) # J17'
par.append([0.2,0.9]) # J17"
par.append([3]) # J18'
par.append([0.6,0.9]) # J18"
par.append([1/3]) # J19
par.append([0.3]) # J20
par.append([0.5]) # J21

incroci = [] # Lista delle informazioni per ciascun incrocio: tipo, strade entranti, strade uscenti, parametri
incroci.append([21,[2,50],[0], par[0]]) # J1
incroci.append([22,[0,32],[3,4], par[1]]) # J2'
incroci.append([12,[5],[1,32], par[2]]) # J2"
incroci.append([21,[4,47],[42], par[3]]) # J3
incroci.append([12,[8],[6,43], par[4]]) # J4
incroci.append([21,[7,43],[44], par[5]]) # J5
incroci.append([12,[44],[45,51], par[6]]) # J6
incroci.append([21,[45,52],[46], par[7]]) # J7
incroci.append([12,[46],[5,47], par[8]]) # J8
incroci.append([21,[8,9],[7], par[9]]) # J9
incroci.append([22,[11,33],[0,10], par[10]]) # J10'
incroci.append([12,[6],[12,33], par[11]]) # J10"
incroci.append([22,[14,15],[11,34], par[12]]) # J11'
incroci.append([21,[12,34],[13], par[13]]) # J11"
incroci.append([22,[17,35],[14,18], par[14]]) # J12'
incroci.append([22,[3,13],[16,35],par[15]]) # J12"
incroci.append([22,[21,53],[20,36], par[16]]) # J13'
incroci.append([22,[19,36],[2,54], par[17]]) # J13"
incroci.append([12,[55],[22,37], par[18]]) # J14'
incroci.append([22,[23,37],[21,56], par[19]]) # J14"
incroci.append([22,[22,38],[24,57], par[20]]) # J15'
incroci.append([12,[25],[23,38], par[21]]) # J15"
incroci.append([12,[58],[23,24], par[22]]) # J15"'
incroci.append([21,[24,39],[26], par[23]]) # J16'
incroci.append([22,[18,27],[25,39], par[24]]) # J16"
incroci.append([12,[26],[28,40], par[25]]) # J17'
incroci.append([22,[29,40],[15,27], par[26]]) # J17"
incroci.append([21,[28,41],[30], par[27]]) # J18'
incroci.append([22,[10,31],[29,41], par[28]]) # J18"
incroci.append([21,[30,49],[59], par[29]]) # J19
incroci.append([12,[60],[8,48], par[30]]) # J20
incroci.append([12,[48],[31,49], par[31]]) # J21

# %%
T = 3600 # Tempo finale
tt = np.arange(0., T, dt) # Vettore dei tempi

n_strade_tot_rete = len(I) + len(j) + len(r)

# Valori delle strade esterne
E_in = E_exp(tt)
E_out = E_cos(tt)

E = np.zeros((11,len(tt)))
E[0,:] = E_sin(tt) # E1
E[1,:] = E_out # E2
E[2,:] = E_in # E3
E[3,:] = E_sin(tt) # E4
E[4,:] = E_out # E5
E[5,:] = E_sin(tt) # E6
E[6,:] = E_out # E7
E[7,:] = E_out # E8
E[8,:] = E_in # E9
E[9,:] = E_out # E10
E[10,:] = E_in # E11

U =[np.zeros(len(xx[i])) for i in range(n_strade_tot_rete)]



# %% [markdown]
# Inseriamo i dati inziali per ciascuna strada principale (tipo I). Per le strade secondarie (j e r) impostiamo che siano vuote al tempo 0.

# %%
rho_0 =[]
rho_0.append(np.ones(len(xx[0]))*0.2) # I1
for i in range(len(xx[0])):
    if xx[0][i] >= 75 and xx[0][i]<= 85:
        rho_0[0][i] = 0.6

rho_0.append(np.ones(len(xx[1]))*0.2) # I2
for i in range(len(xx[1])):
    if xx[1][i] >= 75 and xx[1][i]<= 85:
        rho_0[1][i] = 0.6
rho_0.append((1/425)*xx[2]) # I3
rho_0.append((1/275)*xx[3]) # I4
rho_0.append(0.2*np.ones(len(xx[4]))) # I5
for i in range(len(xx[4])):
    if xx[4][i] >= 420:
        rho_0[4][i] = 0.4

rho_0.append(0.2*np.ones(len(xx[5]))) # I6
rho_0.append(0.2*np.ones(len(xx[6]))) # I7
rho_0.append(0.4*np.ones(len(xx[7]))) # I8
rho_0.append((1/500)*xx[8]) # I9
rho_0.append(0.2*np.ones(len(xx[9]))) # I10
rho_0.append((1/500)*xx[10]) # I11
rho_0.append(0.2*np.ones(len(xx[11]))) # I12
rho_0.append(0.2*np.ones(len(xx[12]))) # I13
rho_0.append(0.2*np.ones(len(xx[13]))) # I14
rho_0.append(0.2*np.ones(len(xx[14]))) # I15
rho_0.append((1/500)*xx[15]) # I16
rho_0.append(np.zeros(len(xx[16]))) # I17
for i in range(len(xx[16])):
    if xx[16][i] >= 75:
        rho_0[16][i] = 0.8

rho_0.append(np.zeros(len(xx[17]))) # I18
rho_0.append(np.ones(len(xx[18]))*0.1) # I19
rho_0.append(np.zeros(len(xx[19]))) # I20
rho_0.append(np.zeros(len(xx[20]))) # I21
for i in range(len(xx[20])):
    if xx[20][i] >= 75:
        rho_0[20][i] = 0.8

rho_0.append((1/500)*xx[21]) # I22
rho_0.append(0.4*np.ones(len(xx[22]))) # I23
rho_0.append(np.zeros(len(xx[23]))) # I24
rho_0.append(np.zeros(len(xx[24]))) # I25
rho_0.append(0.4*np.ones(len(xx[25]))) # I26
rho_0.append(0.1*np.ones(len(xx[26]))) # I27
rho_0.append(0.2*np.ones(len(xx[27]))) # I28
rho_0.append(0.1*np.ones(len(xx[28]))) # I29
rho_0.append(0.2*np.ones(len(xx[29]))) # I30
rho_0.append(0.4*np.ones(len(xx[30]))) # I31
rho_0.append(np.zeros(len(xx[31]))) # I32

for i in range(32,50):
    rho_0.append(np.zeros(len(xx[i])))

# %%
# Aggiungiamo i dati iniziali anche delle strade esterne
temp = rho_0
for i in range(11):
    temp.append([E[i,0]])


# %% [markdown]
# Salviamo le soluzioni solo ad alcuni istanti di tempo, ovvero ogni minuto.

# %%
n_minuti = 60
time_stamps = np.zeros(n_minuti+1)
for i in range(n_minuti):
  time_stamps[i+1] = int(((i+1)*60/dt))

# %%
f = open("sim50", "wb")
dump(tt,f)
dump(time_stamps,f)
counter = 0
for k in range(len(tt)-1):
  # La rete stradale coinvolge due semafori, perciò calcoliamo al tempo tt[j] se sono verdi o rossi
  colore_S1 = semafori(tt[k],1)
  colore_S2 = semafori(tt[k],2)
  if k in time_stamps:
    dump(temp,f)
    counter += 1
    print(counter) # Controllo del progresso
  U = scheme(temp, vectGodunov, dt/dx_min, colore_S1, colore_S2)
  temp = U
  for i in range(11):
    temp.append([E[i,k]])
dump(temp,f)
counter += 1
print(counter)
f.close()



