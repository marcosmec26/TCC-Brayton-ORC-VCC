#

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tespy.networks import Network
from tespy.components import CycleCloser, Compressor, Turbine, Pump, HeatExchanger, SimpleHeatExchanger, MovingBoundaryHeatExchanger
from tespy.connections import Connection
from CoolProp.CoolProp import PropsSI

logging.getLogger("TESPyLogger").setLevel(logging.ERROR)

# 1. REDE

sistema = Network(iterinfo=False)
sistema.units.set_defaults(temperature="degC", temperature_difference="delta_degC", pressure="MPa", pressure_difference="MPa", enthalpy="kJ/kg", power="kW", heat="kW")

# 2. COMPONENTES

cc_b = CycleCloser("fechamento Brayton")
cp_b = Compressor("compressor S-CO2")
rg = HeatExchanger("regenerador")
aq = SimpleHeatExchanger("aquecedor")
tu_b = Turbine("turbina S-CO2")

cc_o = CycleCloser("fechamento ORC")
bo = Pump("bomba ORC")
gv = MovingBoundaryHeatExchanger("gerador de vapor")
tu_o = Turbine("turbina ORC")
co = SimpleHeatExchanger("condensador ORC")

# 3. CONEXÕES

c01 = Connection(cc_b, "out1", cp_b, "in1", label="01")
c02 = Connection(cp_b, "out1", rg, "in2", label="02")
c03 = Connection(rg, "out2", aq, "in1", label="03")
c04 = Connection(aq, "out1", tu_b, "in1", label="04")
c05 = Connection(tu_b, "out1", rg, "in1", label="05")
c06 = Connection(rg, "out1", gv, "in1", label="06")
c07 = Connection(gv, "out1", cc_b, "in1", label="07")

c1 = Connection(cc_o, "out1", bo, "in1", label="1")
c2 = Connection(bo, "out1", gv, "in2", label="2")
c4 = Connection(gv, "out2", tu_o, "in1", label="4")
c5 = Connection(tu_o, "out1", co, "in1", label="5")
c6 = Connection(co, "out1", cc_o, "in1", label="6")

sistema.add_conns(c01, c02, c03, c04, c05, c06, c07, c1, c2, c4, c5, c6)

# 4. PARÂMETROS

FLUIDO_BRAYTON = "CO2"
FLUIDO_ORC = "R245fa"

T_MIN_BRAYTON = 309.15 - 273.15
T_MAX_BRAYTON = 573.15 - 273.15
P_MIN_BRAYTON = 9.183
P_MAX_BRAYTON = 16.352
VAZAO_CO2 = 5.46

ETA_COMPRESSOR = 0.70
ETA_TURBINA_BRAYTON = 0.80
ETA_BOMBA_ORC = 0.70
ETA_TURBINA_ORC = 0.80

T_COND_ORC = 298.15 - 273.15
TD_PINCH = 6.0
DP = 0.1

# 5. DADOS DIGITALIZADOS DE SONG

EPS_TEVAP_SONG = np.array([0.001946, 0.09380, 0.17704, 0.26460, 0.34929, 0.43971, 0.53157, 0.63061, 0.73826, 0.87031, 0.89900])
TEVAP_SONG_K = np.array([306.22, 305.99, 305.80, 305.60, 305.40, 305.20, 304.98, 304.75, 304.50, 304.20, 304.11])

EPS_M_SONG = np.array([0.000386, 0.092311, 0.175280, 0.262190, 0.347780, 0.437330, 0.529510, 0.628280, 0.737580, 0.870590, 0.899560])
M_SONG = np.array([10.8890, 10.1300, 9.4928, 8.8551, 8.2175, 7.5193, 6.8512, 6.0922, 5.2724, 4.2402, 4.0579])

EPS_W_SONG = np.array([0.000000, 0.092308, 0.176470, 0.263350, 0.348870, 0.437100, 0.529410, 0.628510, 0.737100, 0.870140, 0.900000])
W_SONG = np.array([43.629, 39.347, 36.110, 32.768, 29.739, 26.606, 23.055, 19.922, 16.580, 12.820, 11.984])

EPS_ETA_SONG = np.array([0.00000, 0.09345, 0.17684, 0.26310, 0.35080, 0.43994, 0.53051, 0.62971, 0.73898, 0.87268, 0.90000])
ETA_SONG_DECIMAL = np.array([0.020517, 0.019898, 0.019397, 0.018912, 0.018460, 0.017942, 0.017440, 0.016888, 0.016270, 0.015501, 0.015350])

# 6. EFICIÊNCIAS

cp_b.set_attr(eta_s=ETA_COMPRESSOR)
tu_b.set_attr(eta_s=ETA_TURBINA_BRAYTON)
bo.set_attr(eta_s=ETA_BOMBA_ORC)
tu_o.set_attr(eta_s=ETA_TURBINA_ORC)

# 7. GERADOR DE VAPOR

gv.set_attr(dp1=DP, pr2=1.0)

# 8. PRESSÕES DO BRAYTON

P01 = P_MIN_BRAYTON
P02 = P_MAX_BRAYTON
P03 = P02 - DP
P04 = P03 - DP
P07 = P_MIN_BRAYTON
P06 = P07 + DP
P05 = P06 + DP

# 9. ESTADOS FIXOS DO BRAYTON

c01.set_attr(T=T_MIN_BRAYTON, p=P01, m=VAZAO_CO2, fluid={FLUIDO_BRAYTON: 1})
c02.set_attr(p=P02)
c03.set_attr(p=P03)
c04.set_attr(T=T_MAX_BRAYTON, p=P04)
c05.set_attr(p=P05)

# 10. ESTADOS FIXOS DO ORC

c1.set_attr(T=T_COND_ORC, x=0, fluid={FLUIDO_ORC: 1})
c4.set_attr(x=1)

p_condensacao = PropsSI("P", "T", T_COND_ORC + 273.15, "Q", 0, FLUIDO_ORC) / 1e6
c5.set_attr(p=p_condensacao)

# 11. FUNÇÃO PARA CALCULAR h06

def calcular_h06(razao_recuperativa):
    h01 = PropsSI("H", "T", T_MIN_BRAYTON + 273.15, "P", P01 * 1e6, FLUIDO_BRAYTON)
    s01 = PropsSI("S", "T", T_MIN_BRAYTON + 273.15, "P", P01 * 1e6, FLUIDO_BRAYTON)
    h02s = PropsSI("H", "P", P02 * 1e6, "S", s01, FLUIDO_BRAYTON)
    h02 = h01 + (h02s - h01) / ETA_COMPRESSOR
    T02 = PropsSI("T", "P", P02 * 1e6, "H", h02, FLUIDO_BRAYTON)
    h04 = PropsSI("H", "T", T_MAX_BRAYTON + 273.15, "P", P04 * 1e6, FLUIDO_BRAYTON)
    s04 = PropsSI("S", "T", T_MAX_BRAYTON + 273.15, "P", P04 * 1e6, FLUIDO_BRAYTON)
    h05s = PropsSI("H", "P", P05 * 1e6, "S", s04, FLUIDO_BRAYTON)
    h05 = h04 - ETA_TURBINA_BRAYTON * (h04 - h05s)
    h06_limite = PropsSI("H", "T", T02, "P", P06 * 1e6, FLUIDO_BRAYTON)
    h06 = h05 - razao_recuperativa * (h05 - h06_limite)
    return h06 / 1000

def erro_percentual(valor_modelo, valor_referencia):
    return abs((valor_modelo - valor_referencia) / valor_referencia) * 100

# 12. INTERPOLAÇÃO DOS DADOS DO SONG

razoes_recuperativas = np.arange(0.0, 1.0, 0.1)

TEVAP_SONG_INTERP = np.interp(razoes_recuperativas, EPS_TEVAP_SONG, TEVAP_SONG_K)
M_SONG_INTERP = np.interp(razoes_recuperativas, EPS_M_SONG, M_SONG)
W_SONG_INTERP = np.interp(razoes_recuperativas, EPS_W_SONG, W_SONG)
ETA_SONG_INTERP_PERCENT = np.interp(razoes_recuperativas, EPS_ETA_SONG, ETA_SONG_DECIMAL) * 100

# 13. VARREDURA

resultados = []
T_evap_chute = 33.0

print("\n======================================")
print("VALIDAÇÃO COMPLETA BRAYTON + ORC")
print("PINCH = 6 K")
print("======================================\n")

for i, epsilon in enumerate(razoes_recuperativas):

    epsilon = round(float(epsilon), 1)
    h06 = calcular_h06(epsilon)
    c06.set_attr(h=h06)

    # ETAPA 1 - CHUTE DE TEMPERATURA

    c4.set_attr(T=T_evap_chute, x=1)
    gv.set_attr(td_pinch=None)

    try:
        sistema.solve(mode="design", init_previous=True)
    except Exception as erro:
        print(f"ε = {epsilon:.1f} | Etapa inicial não convergiu: {erro}\n")
        continue

    if not sistema.converged:
        print(f"ε = {epsilon:.1f} | Etapa inicial não convergiu.\n")
        continue

    pinch_chute = gv.td_pinch.val

    # ETAPA 2 - T_EVAP LIVRE + PINCH = 6 K

    c4.set_attr(T=None, x=1)
    gv.set_attr(td_pinch=TD_PINCH)

    try:
        sistema.solve(mode="design", init_previous=True)
    except Exception as erro:
        print(f"ε = {epsilon:.1f} | Solução com pinch não convergiu: {erro}\n")
        gv.set_attr(td_pinch=None)
        continue

    if not sistema.converged:
        print(f"ε = {epsilon:.1f} | Solução com pinch não convergiu.\n")
        gv.set_attr(td_pinch=None)
        continue

    # RESULTADOS

    T_evap_tespy_K = c4.T.val + 273.15
    T_evap_chute = c4.T.val

    Q_regenerador = c05.m.val * (c05.h.val - c06.h.val)
    Q_entrada = c03.m.val * (c04.h.val - c03.h.val)
    Q_brayton_orc = c06.m.val * (c06.h.val - c07.h.val)

    W_compressor = c01.m.val * (c02.h.val - c01.h.val)
    W_turbina_brayton = c04.m.val * (c04.h.val - c05.h.val)
    W_liquido_brayton = W_turbina_brayton - W_compressor
    eficiencia_brayton = W_liquido_brayton / Q_entrada

    Q_orc = c4.m.val * (c4.h.val - c2.h.val)
    W_bomba_orc = c1.m.val * (c2.h.val - c1.h.val)
    W_turbina_orc = c4.m.val * (c4.h.val - c5.h.val)
    W_liquido_orc = W_turbina_orc - W_bomba_orc
    eficiencia_orc = W_liquido_orc / Q_orc
    eficiencia_orc_percent = eficiencia_orc * 100

    T_song = TEVAP_SONG_INTERP[i]
    m_song = M_SONG_INTERP[i]
    W_song = W_SONG_INTERP[i]
    eta_song = ETA_SONG_INTERP_PERCENT[i]

    erro_T = erro_percentual(T_evap_tespy_K, T_song)
    erro_m = erro_percentual(c4.m.val, m_song)
    erro_W = erro_percentual(W_liquido_orc, W_song)
    erro_eta = erro_percentual(eficiencia_orc_percent, eta_song)

    diferenca_calor = Q_brayton_orc - Q_orc
    eficiencia_global = (W_liquido_brayton + W_liquido_orc) / Q_entrada

    resultados.append({"Razão recuperativa": epsilon, "T evap TESPy (K)": T_evap_tespy_K, "T evap Song (K)": T_song, "Erro T evap (%)": erro_T, "Pinch chute (K)": pinch_chute, "Pinch final (K)": gv.td_pinch.val, "T06 (K)": c06.T.val + 273.15, "T07 (K)": c07.T.val + 273.15, "p evap ORC (MPa)": c4.p.val, "p condensação ORC (MPa)": c5.p.val, "Vazão TESPy (kg/s)": c4.m.val, "Vazão Song (kg/s)": m_song, "Erro vazão (%)": erro_m, "Potência líquida TESPy (kW)": W_liquido_orc, "Potência líquida Song (kW)": W_song, "Erro potência (%)": erro_W, "Eficiência TESPy (%)": eficiencia_orc_percent, "Eficiência Song (%)": eta_song, "Erro eficiência (%)": erro_eta, "Q Brayton -> ORC (kW)": Q_brayton_orc, "Q absorvido ORC (kW)": Q_orc, "Resíduo térmico (kW)": diferenca_calor, "Potência bomba ORC (kW)": W_bomba_orc, "Potência turbina ORC (kW)": W_turbina_orc, "Potência líquida Brayton (kW)": W_liquido_brayton, "Eficiência Brayton (%)": eficiencia_brayton * 100, "Eficiência global (%)": eficiencia_global * 100})

    print(f"ε = {epsilon:.1f}")
    print(f"T evap TESPy = {T_evap_tespy_K:.3f} K | Song = {T_song:.3f} K | Erro = {erro_T:.3f}%")
    print(f"Pinch inicial = {pinch_chute:.4f} K | Pinch final = {gv.td_pinch.val:.4f} K")
    print(f"p evap = {c4.p.val:.6f} MPa | p cond = {c5.p.val:.6f} MPa")
    print(f"Vazão TESPy = {c4.m.val:.4f} kg/s | Song = {m_song:.4f} kg/s | Erro = {erro_m:.2f}%")
    print(f"Potência líquida TESPy = {W_liquido_orc:.4f} kW | Song = {W_song:.4f} kW | Erro = {erro_W:.2f}%")
    print(f"Eficiência TESPy = {eficiencia_orc_percent:.4f}% | Song = {eta_song:.4f}% | Erro = {erro_eta:.2f}%")
    print(f"Q Brayton -> ORC = {Q_brayton_orc:.4f} kW | Q ORC = {Q_orc:.4f} kW | Resíduo = {diferenca_calor:.8f} kW\n")

    gv.set_attr(td_pinch=None)

# 14. TABELA FINAL

tabela_resultados = pd.DataFrame(resultados)

print("\n======================================")
print("RESULTADOS FINAIS DA VALIDAÇÃO")
print("======================================\n")

print(tabela_resultados.round(4).to_string(index=False))

# 15. MAPE

if not tabela_resultados.empty:

    MAPE_T = tabela_resultados["Erro T evap (%)"].mean()
    MAPE_VAZAO = tabela_resultados["Erro vazão (%)"].mean()
    MAPE_POTENCIA = tabela_resultados["Erro potência (%)"].mean()
    MAPE_EFICIENCIA = tabela_resultados["Erro eficiência (%)"].mean()

    print("\n======================================")
    print("ERROS MÉDIOS DA VALIDAÇÃO")
    print("======================================\n")
    print(f"MAPE temperatura de evaporação = {MAPE_T:.3f}%")
    print(f"MAPE vazão mássica = {MAPE_VAZAO:.3f}%")
    print(f"MAPE potência líquida = {MAPE_POTENCIA:.3f}%")
    print(f"MAPE eficiência ORC = {MAPE_EFICIENCIA:.3f}%")

# 16. GRÁFICO - TEMPERATURA DE EVAPORAÇÃO + DESVIO RELATIVO

if not tabela_resultados.empty:

    epsilon = tabela_resultados["Razão recuperativa"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    linha1, = ax1.plot(epsilon, tabela_resultados["T evap TESPy (K)"], marker="o", label="Este trabalho")
    linha2, = ax1.plot(epsilon, tabela_resultados["T evap Song (K)"], marker="s", label="Song")
    linha3, = ax2.plot(epsilon, tabela_resultados["Erro T evap (%)"], marker="d", linestyle="--", label="Desvio relativo")

    ax1.set_xlabel(r"Efetividade do regenerador, $\varepsilon_B$")
    ax1.set_ylabel("Temperatura de evaporação do ORC (K)")
    ax2.set_ylabel("Desvio relativo (%)")
    ax1.set_title("Verificação da temperatura de evaporação do ORC")
    ax1.set_xticks(epsilon)
    ax1.grid()
    ax1.legend(handles=[linha1, linha2, linha3], loc="best")

    plt.tight_layout()
    plt.show()

# 17. GRÁFICO - VAZÃO MÁSSICA + DESVIO RELATIVO

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    linha1, = ax1.plot(epsilon, tabela_resultados["Vazão TESPy (kg/s)"], marker="o", label="Este trabalho")
    linha2, = ax1.plot(epsilon, tabela_resultados["Vazão Song (kg/s)"], marker="s", label="Song")
    linha3, = ax2.plot(epsilon, tabela_resultados["Erro vazão (%)"], marker="d", linestyle="--", label="Desvio relativo")

    ax1.set_xlabel(r"Efetividade do regenerador, $\varepsilon_B$")
    ax1.set_ylabel("Vazão mássica do R245fa (kg/s)")
    ax2.set_ylabel("Desvio relativo (%)")
    ax1.set_title("Verificação da vazão mássica do ORC")
    ax1.set_xticks(epsilon)
    ax1.grid()
    ax1.legend(handles=[linha1, linha2, linha3], loc="best")

    plt.tight_layout()
    plt.show()

# 18. GRÁFICO - POTÊNCIA LÍQUIDA + DESVIO RELATIVO

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    linha1, = ax1.plot(epsilon, tabela_resultados["Potência líquida TESPy (kW)"], marker="o", label="Este trabalho")
    linha2, = ax1.plot(epsilon, tabela_resultados["Potência líquida Song (kW)"], marker="s", label="Song")
    linha3, = ax2.plot(epsilon, tabela_resultados["Erro potência (%)"], marker="d", linestyle="--", label="Desvio relativo")

    ax1.set_xlabel(r"Efetividade do regenerador, $\varepsilon_B$")
    ax1.set_ylabel("Potência líquida do ORC (kW)")
    ax2.set_ylabel("Desvio relativo (%)")
    ax1.set_title("Verificação da potência líquida do ORC")
    ax1.set_xticks(epsilon)
    ax1.grid()
    ax1.legend(handles=[linha1, linha2, linha3], loc="best")

    plt.tight_layout()
    plt.show()

# 19. GRÁFICO - EFICIÊNCIA TÉRMICA + DESVIO RELATIVO

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    linha1, = ax1.plot(epsilon, tabela_resultados["Eficiência TESPy (%)"], marker="o", label="Este trabalho")
    linha2, = ax1.plot(epsilon, tabela_resultados["Eficiência Song (%)"], marker="s", label="Song")
    linha3, = ax2.plot(epsilon, tabela_resultados["Erro eficiência (%)"], marker="d", linestyle="--", label="Desvio relativo")

    ax1.set_xlabel(r"Efetividade do regenerador, $\varepsilon_B$")
    ax1.set_ylabel("Eficiência térmica do ORC (%)")
    ax2.set_ylabel("Desvio relativo (%)")
    ax1.set_title("Verificação da eficiência térmica do ORC")
    ax1.set_xticks(epsilon)
    ax1.grid()
    ax1.legend(handles=[linha1, linha2, linha3], loc="best")

    plt.tight_layout()
    plt.show()
