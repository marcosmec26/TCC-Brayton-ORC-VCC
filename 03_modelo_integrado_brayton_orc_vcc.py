#
#

import logging
import numpy as np
import pandas as pd

from CoolProp.CoolProp import PropsSI

from tespy.networks import Network
from tespy.components import CycleCloser, Compressor, Turbine, Pump, HeatExchanger, SimpleHeatExchanger, MovingBoundaryHeatExchanger, Splitter, Merge, Valve, Source, Sink, PowerBus, PowerSink, PowerSource
from tespy.connections import Connection, PowerConnection

from exerpy import ExergyAnalysis

logging.getLogger("TESPyLogger").setLevel(logging.ERROR)
logging.getLogger("exerpy").setLevel(logging.ERROR)

# 1. FLUIDOS

FLUIDO_BRAYTON = "CO2"

FLUIDOS = ["R600", "R600a", "R1234ze(E)", "R602"]

def nome_coolprop(fluido):
    if fluido == "R602":
        return "n-Hexane"
    return fluido

# 2. CONFIGURAÇÕES DO ORC

CONFIGURACOES_ORC = ["Sem recuperador", "Com recuperador"]

# 3. PARÂMETROS GERAIS

T_MIN_BRAYTON_K = 309.15
T_MAX_BRAYTON_K = 573.15

T_MIN_BRAYTON_C = T_MIN_BRAYTON_K - 273.15
T_MAX_BRAYTON_C = T_MAX_BRAYTON_K - 273.15

P_MIN_BRAYTON = 9.183
P_MAX_BRAYTON = 16.352
VAZAO_CO2 = 5.46

# Gases de exaustão - composição em fração mássica
GASES_EXAUSTAO = {
    "N2": 0.72990,
    "H2O": 0.10827,
    "CO2": 0.13304,
    "O2": 0.02879
}

T_GASES_ENTRADA = 600.0
T_GASES_SAIDA_REFERENCIA = 180.0
P_GASES = 0.101325

ETA_COMPRESSOR_B = 0.70
ETA_TURBINA_B = 0.80

ETA_BOMBA_ORC = 0.80
ETA_TURBINA_ORC = 0.80

ETA_COMPRESSOR_VCC = 0.75

T_COND = 25.0
T_EVAP_VCC = 0.0

# Vazão do VCC calculada pela potência disponível na turbina ORC

# Corrente auxiliar de ar seco no evaporador
T_AR_ENTRADA = 25.0
T_AR_SAIDA_CHUTE = 10.0
P_AR = 0.101325

TD_PINCH = 6.0
TD_PINCH_GASES = 6.0
TD_PINCH_EVAP = 6.0

T_EVAP_CHUTE_INICIAL = 33.0

# 4. ESTADO DE REFERÊNCIA PARA A EXERGIA

T0 = 25.0 + 273.15
P0 = 101325

# 5. PRESSÕES DO BRAYTON

P01 = P_MIN_BRAYTON
P02 = P_MAX_BRAYTON

# 6. FUNÇÕES DE PROPRIEDADES

def calcular_h_ps(fluido, pressao_mpa, entropia):
    return PropsSI("H", "P", pressao_mpa * 1e6, "S", entropia, fluido) / 1000.0

def calcular_h_tp(fluido, temperatura_K, pressao_mpa):
    return PropsSI("H", "T", temperatura_K, "P", pressao_mpa * 1e6, fluido) / 1000.0

def calcular_s_tp(fluido, temperatura_K, pressao_mpa):
    return PropsSI("S", "T", temperatura_K, "P", pressao_mpa * 1e6, fluido)

# 7. ENTALPIA NA SAÍDA FRIA DO REGENERADOR DO BRAYTON

def calcular_h03_brayton(epsilon_b, perda_pressao):

    pr = 1 - perda_pressao

    P03 = P02 * pr
    P04 = P03 * pr

    P07 = P01
    P06 = P07 / pr
    P05 = P06 / pr

    h01 = calcular_h_tp(FLUIDO_BRAYTON, T_MIN_BRAYTON_K, P01)
    s01 = calcular_s_tp(FLUIDO_BRAYTON, T_MIN_BRAYTON_K, P01)

    h02s = calcular_h_ps(FLUIDO_BRAYTON, P02, s01)
    h02 = h01 + (h02s - h01) / ETA_COMPRESSOR_B

    h04 = calcular_h_tp(FLUIDO_BRAYTON, T_MAX_BRAYTON_K, P04)
    s04 = calcular_s_tp(FLUIDO_BRAYTON, T_MAX_BRAYTON_K, P04)

    h05s = calcular_h_ps(FLUIDO_BRAYTON, P05, s04)
    h05 = h04 - ETA_TURBINA_B * (h04 - h05s)

    h03 = h02 + epsilon_b * (h05 - h02)

    return h03

# 8. FUNÇÃO PRINCIPAL

def resolver_caso(fluido, epsilon_b, perda_pressao, configuracao_orc="Sem recuperador", epsilon_orc=0.0, T_chute=None):

    if T_chute is None:
        T_chute = T_EVAP_CHUTE_INICIAL

    pr = 1 - perda_pressao
    fluido_cp = nome_coolprop(fluido)

    # 8.1 REDE

    nw = Network(iterinfo=False)

    nw.units.set_defaults(
        temperature="degC",
        temperature_difference="delta_degC",
        pressure="MPa",
        pressure_difference="MPa",
        enthalpy="kJ/kg",
        power="kW",
        heat="kW"
    )

    # 8.2 COMPONENTES DO BRAYTON

    fechamento_brayton = CycleCloser("fechamento Brayton")
    compressor_brayton = Compressor("compressor Brayton")
    regenerador_brayton = HeatExchanger("regenerador Brayton")
    aquecedor_brayton = MovingBoundaryHeatExchanger("trocador gases-CO2")
    entrada_gases = Source("entrada gases de exaustão")
    saida_gases = Sink("saída gases de exaustão")
    turbina_brayton = Turbine("turbina Brayton")
    gerador_vapor = MovingBoundaryHeatExchanger("gerador de vapor")

    # 8.3 COMPONENTES DO ORC

    bomba_orc = Pump("bomba ORC")
    turbina_orc = Turbine("turbina ORC")

    if configuracao_orc == "Com recuperador":
        recuperador_orc = HeatExchanger("recuperador ORC")

    # 8.4 COMPONENTES DO VCC

    valvula_vcc = Valve("válvula de expansão VCC")
    evaporador_vcc = MovingBoundaryHeatExchanger("evaporador VCC")
    entrada_ar = Source("entrada ar evaporador")
    saida_ar = Sink("saída ar evaporador")
    compressor_vcc = Compressor("compressor VCC")

    misturador = Merge("misturador")
    fechamento_inferior = CycleCloser("fechamento ciclo inferior")
    condensador = SimpleHeatExchanger("condensador compartilhado")
    separador = Splitter("separador")

    # 8.5 COMPONENTES DE POTÊNCIA

    eixo_brayton = PowerBus("eixo Brayton", num_in=1, num_out=2)
    potencia_brayton = PowerSink("potência líquida Brayton")

    eixo_orc_vcc = PowerBus("eixo turbina ORC-VCC", num_in=1, num_out=1)
    fonte_bomba_orc = PowerSource("fonte elétrica bomba ORC")

    # 8.6 CONEXÕES DO BRAYTON

    c01 = Connection(fechamento_brayton, "out1", compressor_brayton, "in1", label="01")
    c02 = Connection(compressor_brayton, "out1", regenerador_brayton, "in2", label="02")
    c03 = Connection(regenerador_brayton, "out2", aquecedor_brayton, "in2", label="03")
    c04 = Connection(aquecedor_brayton, "out2", turbina_brayton, "in1", label="04")
    c05 = Connection(turbina_brayton, "out1", regenerador_brayton, "in1", label="05")

    c_gas_in = Connection(entrada_gases, "out1", aquecedor_brayton, "in1", label="gases_in")
    c_gas_out = Connection(aquecedor_brayton, "out1", saida_gases, "in1", label="gases_out")

    c06 = Connection(regenerador_brayton, "out1", gerador_vapor, "in1", label="06")
    c07 = Connection(gerador_vapor, "out1", fechamento_brayton, "in1", label="07")

    # 8.7 CONDENSADOR COMPARTILHADO

    c_mix = Connection(misturador, "out1", fechamento_inferior, "in1", label="mix")
    c_cond_in = Connection(fechamento_inferior, "out1", condensador, "in1", label="cond_in")
    c7 = Connection(condensador, "out1", separador, "in1", label="7=1")

    # 8.8 ORC

    c1 = Connection(separador, "out1", bomba_orc, "in1", label="1")

    if configuracao_orc == "Sem recuperador":

        c2 = Connection(bomba_orc, "out1", gerador_vapor, "in2", label="2")
        c4 = Connection(gerador_vapor, "out2", turbina_orc, "in1", label="4")
        c5 = Connection(turbina_orc, "out1", misturador, "in1", label="5")

    else:

        c2 = Connection(bomba_orc, "out1", recuperador_orc, "in2", label="2")
        c3 = Connection(recuperador_orc, "out2", gerador_vapor, "in2", label="3")
        c4 = Connection(gerador_vapor, "out2", turbina_orc, "in1", label="4")
        c5 = Connection(turbina_orc, "out1", recuperador_orc, "in1", label="5")
        c6 = Connection(recuperador_orc, "out1", misturador, "in1", label="6")

    # 8.9 VCC

    # Lado 1 do evaporador: ar seco (corrente quente)
    c_ar_in = Connection(entrada_ar, "out1", evaporador_vcc, "in1", label="ar_in")
    c_ar_out = Connection(evaporador_vcc, "out1", saida_ar, "in1", label="ar_out")

    # Lado 2 do evaporador: refrigerante do VCC (corrente fria)
    c001 = Connection(separador, "out2", valvula_vcc, "in1", label="001")
    c002 = Connection(valvula_vcc, "out1", evaporador_vcc, "in2", label="002")
    c003 = Connection(evaporador_vcc, "out2", compressor_vcc, "in1", label="003")
    c004 = Connection(compressor_vcc, "out1", misturador, "in2", label="004")

    # 8.10 CONEXÕES DE POTÊNCIA

    # Brayton
    p_turb_b = PowerConnection(turbina_brayton, "power", eixo_brayton, "power_in1", label="P_turbina_B")
    p_comp_b = PowerConnection(eixo_brayton, "power_out1", compressor_brayton, "power", label="P_compressor_B")
    p_liq_b = PowerConnection(eixo_brayton, "power_out2", potencia_brayton, "power", label="P_liquida_B")

    # ORC
    p_turb_orc = PowerConnection(turbina_orc, "power", eixo_orc_vcc, "power_in1", label="P_turbina_ORC")
    p_comp_vcc = PowerConnection(eixo_orc_vcc, "power_out1", compressor_vcc, "power", label="P_compressor_VCC")
    p_bomba_orc = PowerConnection(fonte_bomba_orc, "power", bomba_orc, "power", label="P_bomba_ORC")

    # 8.11 ADICIONA AS CONEXÕES

    if configuracao_orc == "Sem recuperador":
        nw.add_conns(
            c01, c02, c03, c04, c05, c06, c07,
            c_gas_in, c_gas_out,
            c_mix, c_cond_in, c7,
            c1, c2, c4, c5,
            c001, c002, c003, c004,
            c_ar_in, c_ar_out
        )

    else:
        nw.add_conns(
            c01, c02, c03, c04, c05, c06, c07,
            c_gas_in, c_gas_out,
            c_mix, c_cond_in, c7,
            c1, c2, c3, c4, c5, c6,
            c001, c002, c003, c004,
            c_ar_in, c_ar_out
        )

    nw.add_conns(
        p_turb_b,
        p_comp_b,
        p_liq_b,
        p_turb_orc,
        p_bomba_orc,
        p_comp_vcc
    )

    # 9. PARÂMETROS DOS COMPONENTES

    compressor_brayton.set_attr(eta_s=ETA_COMPRESSOR_B)
    turbina_brayton.set_attr(eta_s=ETA_TURBINA_B)

    regenerador_brayton.set_attr(pr1=pr, pr2=pr)
    aquecedor_brayton.set_attr(pr1=1.0, pr2=pr)

    gerador_vapor.set_attr(pr1=pr, pr2=pr)

    bomba_orc.set_attr(eta_s=ETA_BOMBA_ORC)
    turbina_orc.set_attr(eta_s=ETA_TURBINA_ORC)

    if configuracao_orc == "Com recuperador":
        recuperador_orc.set_attr(pr1=pr, pr2=pr, eff_cold=epsilon_orc)

    condensador.set_attr(pr=pr)

    # Ar externo sem perda de pressão; refrigerante com a perda da análise paramétrica
    evaporador_vcc.set_attr(pr1=1.0, pr2=pr)

    compressor_vcc.set_attr(eta_s=ETA_COMPRESSOR_VCC)

    # 10. CONDIÇÕES DO BRAYTON

    c01.set_attr(T=T_MIN_BRAYTON_C, p=P01, m=VAZAO_CO2, fluid={FLUIDO_BRAYTON: 1})
    c02.set_attr(p=P02)
    c03.set_attr(h=calcular_h03_brayton(epsilon_b, perda_pressao))
    c04.set_attr(T=T_MAX_BRAYTON_C)

    c_gas_in.set_attr(T=T_GASES_ENTRADA, p=P_GASES, fluid=GASES_EXAUSTAO)

    # 180 °C é apenas um chute inicial
    c_gas_out.set_attr(T=T_GASES_SAIDA_REFERENCIA)

    # 11. CONDIÇÕES DO CICLO INFERIOR

    c7.set_attr(T=T_COND, x=0, fluid={fluido_cp: 1})
    c4.set_attr(x=1)

    c003.set_attr(T=T_EVAP_VCC, x=1)
    # Vazão do VCC livre: calculada pelo acoplamento de potência turbina ORC -> compressor VCC

    # O ar entra no estado térmico de referência da análise exergética.
    # A vazão de ar não é fixada: o TESPy a determina junto com a
    # temperatura de saída quando o pinch do evaporador é imposto.
    c_ar_in.set_attr(T=T_AR_ENTRADA, p=P_AR, fluid={"air": 1})

    # 12. ETAPA 1 - CHUTE INICIAL

    c4.set_attr(T=T_chute, x=1)

    gerador_vapor.set_attr(td_pinch=None)
    aquecedor_brayton.set_attr(td_pinch=None)

    # Na primeira etapa é usado apenas um chute para a saída do ar.
    # Isso fornece um ponto inicial para a vazão de ar antes de impor o pinch.
    evaporador_vcc.set_attr(td_pinch=None)
    c_ar_out.set_attr(T=T_AR_SAIDA_CHUTE)

    nw.solve(mode="design", init_previous=True)

    if not nw.converged:
        raise RuntimeError("A etapa inicial não convergiu.")

    pinch_inicial = gerador_vapor.td_pinch.val
    pinch_gases_inicial = aquecedor_brayton.td_pinch.val

    # 13. ETAPA 2 - PINCH DE 6 K

    c4.set_attr(T=None, x=1)
    c_gas_out.set_attr(T=None)
    c_ar_out.set_attr(T=None)

    gerador_vapor.set_attr(td_pinch=TD_PINCH)
    aquecedor_brayton.set_attr(td_pinch=TD_PINCH_GASES)
    evaporador_vcc.set_attr(td_pinch=TD_PINCH_EVAP)

    nw.solve(mode="design", init_previous=True)

    if not nw.converged:
        raise RuntimeError("A solução final com pinch de 6 K não convergiu.")

    pinch_gases_final = aquecedor_brayton.td_pinch.val
    pinch_evap_final = evaporador_vcc.td_pinch.val
    T_gases_saida = c_gas_out.T.val
    T_ar_saida = c_ar_out.T.val
    m_ar = c_ar_in.m.val

    if pinch_gases_final < TD_PINCH_GASES - 1e-4:
        raise RuntimeError("O pinch gases-CO2 não foi atendido.")

    if pinch_evap_final < TD_PINCH_EVAP - 1e-4:
        raise RuntimeError("O pinch do evaporador VCC não foi atendido.")

    if T_gases_saida >= T_GASES_SAIDA_REFERENCIA:
        limite_gases = "Sim"
    else:
        limite_gases = "Não"

    # 14. TEMPERATURA DE EVAPORAÇÃO DO ORC

    T_evap_orc = c4.T.val

    if T_evap_orc <= T_COND:
        raise RuntimeError("A temperatura de evaporação do ORC ficou abaixo ou igual à temperatura de condensação.")

    Tcrit = PropsSI("Tcrit", fluido_cp) - 273.15

    if T_evap_orc >= Tcrit:
        raise RuntimeError("A temperatura de evaporação atingiu ou ultrapassou a temperatura crítica.")

    # 15. BALANÇO ENERGÉTICO DO BRAYTON

    W_comp_b = c01.m.val * (c02.h.val - c01.h.val)
    W_turb_b = c04.m.val * (c04.h.val - c05.h.val)
    W_liq_b = W_turb_b - W_comp_b

    Q_in_b = c03.m.val * (c04.h.val - c03.h.val)
    Q_gases = c_gas_in.m.val * (c_gas_in.h.val - c_gas_out.h.val)
    Q_brayton_orc = c06.m.val * (c06.h.val - c07.h.val)

    eta_b = W_liq_b / Q_in_b

    erro_balanco_aquecedor = 100.0 * abs(Q_gases - Q_in_b) / max(abs(Q_in_b), 1e-9)

    # 16. ORC

    m_orc = c1.m.val

    T_entrada_fria_rec = c2.T.val
    T_entrada_quente_rec = c5.T.val

    deltaT_disponivel_rec = T_entrada_quente_rec - T_entrada_fria_rec

    if configuracao_orc == "Sem recuperador":

        h3_orc = c2.h.val

        Q_rec_frio = 0.0
        Q_rec_quente = 0.0
        Q_rec_orc = 0.0
        erro_balanco_rec = 0.0

        recuperacao_possivel = "Sim" if deltaT_disponivel_rec > 0 else "Não"

    else:

        if deltaT_disponivel_rec <= 0:
            raise RuntimeError("Recuperador ORC inviável: não existe diferença de temperatura favorável.")

        h3_orc = c3.h.val

        Q_rec_frio = m_orc * (c3.h.val - c2.h.val)
        Q_rec_quente = m_orc * (c5.h.val - c6.h.val)
        Q_rec_orc = Q_rec_frio

        if Q_rec_frio <= 0:
            raise RuntimeError("Recuperador ORC inviável: calor no lado frio ficou nulo ou negativo.")

        if Q_rec_quente <= 0:
            raise RuntimeError("Recuperador ORC inviável: calor no lado quente ficou nulo ou negativo.")

        if c5.T.val <= c3.T.val:
            raise RuntimeError("Recuperador ORC inviável: ocorreu cruzamento de temperatura.")

        if c6.T.val <= c2.T.val:
            raise RuntimeError("Recuperador ORC inviável: ocorreu cruzamento de temperatura.")

        erro_balanco_rec = 100.0 * abs(Q_rec_frio - Q_rec_quente) / max(abs(Q_rec_frio), 1e-9)

        recuperacao_possivel = "Sim"

    Q_gv_orc = m_orc * (c4.h.val - h3_orc)

    W_bomba_orc = m_orc * (c2.h.val - c1.h.val)
    W_turbina_orc = m_orc * (c4.h.val - c5.h.val)
    W_liq_orc = W_turbina_orc - W_bomba_orc

    eta_orc = W_liq_orc / Q_gv_orc

    erro_balanco_gv = 100.0 * abs(Q_brayton_orc - Q_gv_orc) / max(abs(Q_brayton_orc), 1e-9)

    # 17. VCC

    m_vcc = c001.m.val

    W_comp_vcc = m_vcc * (c004.h.val - c003.h.val)
    Q_evap_vcc = m_vcc * (c003.h.val - c002.h.val)

    Q_ar = m_ar * (c_ar_in.h.val - c_ar_out.h.val)
    erro_balanco_evap = 100.0 * abs(Q_evap_vcc - Q_ar) / max(abs(Q_evap_vcc), 1e-9)

    COP_vcc = Q_evap_vcc / W_comp_vcc

    atendimento = 100.0 * W_turbina_orc / W_comp_vcc
    saldo = W_turbina_orc - W_comp_vcc
    erro_acoplamento = 100.0 * abs(saldo) / max(abs(W_turbina_orc), 1e-9)
    if W_turbina_orc <= 0 or m_vcc <= 0:
        raise RuntimeError("Acoplamento ORC-VCC sem potência positiva da turbina ou vazão positiva do VCC.")
    if erro_acoplamento > 1e-4:
        raise RuntimeError("O balanço de potência entre turbina ORC e compressor VCC não fechou.")

    # 18. RESULTADOS ENERGÉTICOS

    resultado = {
        "Fluido": fluido,
        "Configuração ORC": configuracao_orc,
        "epsilon Brayton": epsilon_b,
        "epsilon recuperador ORC": epsilon_orc,
        "Perda de pressão (%)": perda_pressao * 100.0,
        "T evap ORC (°C)": T_evap_orc,
        "Pinch final GV (K)": gerador_vapor.td_pinch.val,
        "Pinch final gases-CO2 (K)": pinch_gases_final,
        "Pinch final evaporador VCC (K)": pinch_evap_final,
        "T crítica fluido (°C)": Tcrit,
        "Margem crítica (K)": Tcrit - T_evap_orc,
        "m gases de exaustão (kg/s)": c_gas_in.m.val,
        "T gases saída (°C)": T_gases_saida,
        "Limite 180 °C atendido?": limite_gases,
        "Q gases->Brayton (kW)": Q_gases,
        "Q entrada Brayton (kW)": Q_in_b,
        "Erro balanço aquecedor (%)": erro_balanco_aquecedor,
        "m ORC (kg/s)": m_orc,
        "Delta T disponível recuperador (K)": deltaT_disponivel_rec,
        "Recuperação possível?": recuperacao_possivel,
        "Q Brayton->ORC (kW)": Q_brayton_orc,
        "Q absorvido ORC (kW)": Q_gv_orc,
        "Q recuperador ORC (kW)": Q_rec_orc,
        "Erro balanço recuperador (%)": erro_balanco_rec,
        "Erro balanço GV (%)": erro_balanco_gv,
        "W compressor Brayton (kW)": W_comp_b,
        "W turbina Brayton (kW)": W_turb_b,
        "W líquido Brayton (kW)": W_liq_b,
        "Eficiência Brayton (%)": eta_b * 100.0,
        "W bomba ORC (kW)": W_bomba_orc,
        "W turbina ORC (kW)": W_turbina_orc,
        "W líquido ORC (kW)": W_liq_orc,
        "Eficiência ORC (%)": eta_orc * 100.0,
        "W compressor VCC (kW)": W_comp_vcc,
        "Q evaporador VCC (kW)": Q_evap_vcc,
        "m ar evaporador (kg/s)": m_ar,
        "T ar entrada (°C)": c_ar_in.T.val,
        "T ar saída (°C)": T_ar_saida,
        "Q retirado do ar (kW)": Q_ar,
        "Erro balanço evaporador (%)": erro_balanco_evap,
        "COP VCC": COP_vcc,
        "Atendimento VCC (%)": atendimento,
        "Saldo potência (kW)": saldo,
        "Erro acoplamento turbina ORC-VCC (%)": erro_acoplamento,
        "m VCC (kg/s)": m_vcc
    }

    # 19. IMPORTA A REDE PARA O EXERPY

    ean = ExergyAnalysis.from_tespy(
        nw,
        Tamb=T0,
        pamb=P0,
        split_physical_exergy=True
    )

    # 20. DEFINIÇÃO GLOBAL DE FUEL, PRODUTO E PERDA
    #
    # A turbina ORC alimenta internamente o compressor VCC.
    # A bomba ORC recebe trabalho externo independente.
    # Os gases de saída são perda exergética, não subtração do fuel.
    # O condensador também rejeita exergia não aproveitada.

    fuel = {
        "inputs": ["gases_in", "P_bomba_ORC"]
    }

    produto = {
        "inputs": ["P_liquida_B", "ar_out"],
        "outputs": ["ar_in"]
    }

    perda = {
        "outputs": ["gases_out"]
    }

    ean.analyse(
        E_F=fuel,
        E_P=produto,
        E_L=perda
    )

    # 21. EXERGIAS DAS CORRENTES EXTERNAS

    E_gases_in = ean.connections["gases_in"]["E"] / 1000.0
    E_gases_out = ean.connections["gases_out"]["E"] / 1000.0

    E_ar_in = ean.connections["ar_in"]["E"] / 1000.0
    E_ar_out = ean.connections["ar_out"]["E"] / 1000.0
    E_produto_frio = E_ar_out - E_ar_in

    E_potencia_vcc = ean.connections["P_compressor_VCC"]["E"] / 1000.0
    E_potencia_b = ean.connections["P_liquida_B"]["E"] / 1000.0
    E_potencia_orc = ean.connections["P_turbina_ORC"]["E"] / 1000.0
    E_potencia_bomba_orc = ean.connections["P_bomba_ORC"]["E"] / 1000.0

    # 22. RESULTADOS DO EVAPORADOR NO EXERPY

    evaporador_exerpy = ean.components["evaporador VCC"]

    E_F_evap = evaporador_exerpy.E_F / 1000.0
    E_P_evap = evaporador_exerpy.E_P / 1000.0
    E_D_evap = evaporador_exerpy.E_D / 1000.0
    eta_ex_evap = evaporador_exerpy.epsilon

    # 23. BALANÇO EXERGÉTICO GLOBAL

    # O ExerPy calcula o condensador como um componente com E_F, E_P e E_D.
    # Como a corrente externa que recebe o calor do condensador não foi
    # explicitamente modelada, o E_P do condensador representa a exergia
    # rejeitada para fora da fronteira do sistema. No balanço global, essa
    # parcela é classificada como perda exergética (E_L).

    condensador_exerpy = ean.components["condensador compartilhado"]
    E_L_condensador = condensador_exerpy.E_P / 1000.0

    # A exergia dos gases de saída e a rejeição no condensador são perdas.
    # A potência da bomba ORC é fornecida externamente.
    E_F_sistema = E_gases_in + E_potencia_bomba_orc
    E_P_sistema = E_potencia_b + E_produto_frio
    E_L_sistema = E_gases_out + E_L_condensador
    E_D_global = E_F_sistema - E_P_sistema - E_L_sistema
    eta_ex_global = E_P_sistema / E_F_sistema

    E_D_componentes = 0.0

    for componente in ean.components.values():
        if hasattr(componente, "E_D"):
            if componente.E_D is not None and not np.isnan(componente.E_D):
                E_D_componentes += componente.E_D / 1000.0

    diferenca_balanco_exergia = E_D_global - E_D_componentes
    erro_balanco_exergia = 100.0 * abs(diferenca_balanco_exergia) / max(abs(E_F_sistema), 1e-9)

    # 24. ADICIONA RESULTADOS EXERGÉTICOS

    resultado["E gases entrada (kW)"] = E_gases_in
    resultado["E gases saída (kW)"] = E_gases_out
    resultado["E ar entrada (kW)"] = E_ar_in
    resultado["E ar saída (kW)"] = E_ar_out
    resultado["E produto frigorífico (kW)"] = E_produto_frio

    resultado["E potência compressor VCC (kW)"] = E_potencia_vcc
    resultado["E potência líquida Brayton (kW)"] = E_potencia_b
    resultado["E potência turbina ORC (kW)"] = E_potencia_orc
    resultado["E potência bomba ORC externa (kW)"] = E_potencia_bomba_orc
    resultado["E_L gases de exaustão (kW)"] = E_gases_out

    resultado["E_F sistema (kW)"] = E_F_sistema
    resultado["E_P sistema (kW)"] = E_P_sistema
    resultado["E_L condensador (kW)"] = E_L_condensador
    resultado["E_L sistema (kW)"] = E_L_sistema
    resultado["E_D global (kW)"] = E_D_global
    resultado["Eficiência exergética global (%)"] = eta_ex_global * 100.0
    resultado["Erro acoplamento turbina ORC-VCC (%)"] = erro_acoplamento

    resultado["E_F evaporador VCC (kW)"] = E_F_evap
    resultado["E_P evaporador VCC (kW)"] = E_P_evap
    resultado["E_D evaporador VCC (kW)"] = E_D_evap
    resultado["Eficiência exergética evaporador VCC (%)"] = eta_ex_evap * 100.0

    resultado["Soma E_D componentes (kW)"] = E_D_componentes
    resultado["Diferença balanço exergético (kW)"] = diferenca_balanco_exergia
    resultado["Erro balanço exergético (%)"] = erro_balanco_exergia

    # 25. RESULTADOS EXERGÉTICOS POR COMPONENTE

    for nome, componente in ean.components.items():

        if hasattr(componente, "E_D") and componente.E_D is not None:
            if not np.isnan(componente.E_D):
                resultado[f"E_D {nome} (kW)"] = componente.E_D / 1000.0

        if hasattr(componente, "epsilon") and componente.epsilon is not None:
            if not np.isnan(componente.epsilon):
                resultado[f"Eficiência exergética {nome} (%)"] = componente.epsilon * 100.0

    return resultado, nw, ean

# 26. VARREDURA FINAL - 2.000 CASOS

EPSILONS_BRAYTON = np.round(np.arange(0.0, 1.0, 0.1), 1)
EPSILONS_ORC = np.round(np.arange(0.0, 1.0, 0.1), 1)
PERDAS_PRESSAO = [0.01, 0.02, 0.03, 0.04, 0.05]

resultados = []

total_casos = (
    len(FLUIDOS)
    * len(PERDAS_PRESSAO)
    * len(EPSILONS_BRAYTON)
    * len(EPSILONS_ORC)
)

contador = 0
solucoes_encontradas = 0
recuperacoes_inviaveis = 0
casos_sem_solucao_fisica = 0
casos_sem_solucao = 0

print("\n" + "=" * 100)
print("INÍCIO DA VARREDURA FINAL ENERGÉTICA + EXERGÉTICA")
print("=" * 100)
print(f"Número total de combinações = {total_casos}")
print("=" * 100)

for fluido in FLUIDOS:

    print("\n" + "=" * 100)
    print(f"FLUIDO = {fluido}")
    print("=" * 100)

    for perda_pressao in PERDAS_PRESSAO:

        print(f"\nPERDA DE PRESSÃO = {perda_pressao * 100:.0f}%")

        for epsilon_b in EPSILONS_BRAYTON:

            T_chute = T_EVAP_CHUTE_INICIAL

            for epsilon_orc in EPSILONS_ORC:

                contador += 1

                if np.isclose(epsilon_orc, 0.0):
                    configuracao_orc = "Sem recuperador"
                else:
                    configuracao_orc = "Com recuperador"

                try:

                    resultado, _, _ = resolver_caso(
                        fluido,
                        float(epsilon_b),
                        perda_pressao,
                        configuracao_orc,
                        float(epsilon_orc),
                        T_chute
                    )

                    # Guarda a temperatura convergida como chute para o caso seguinte.
                    T_chute = resultado["T evap ORC (°C)"]

                    resultado["Status"] = "Solução encontrada"
                    resultado["Mensagem"] = ""

                    resultados.append(resultado)
                    solucoes_encontradas += 1

                    print(
                        f"[{contador}/{total_casos}] "
                        f"{fluido} | perda = {perda_pressao * 100:.0f}% | "
                        f"epsilon Brayton = {epsilon_b:.1f} | "
                        f"epsilon ORC = {epsilon_orc:.1f} | "
                        f"T evap = {resultado['T evap ORC (°C)']:.3f} °C | "
                        f"T gases = {resultado['T gases saída (°C)']:.2f} °C | "
                        f"eta ex = {resultado['Eficiência exergética global (%)']:.3f}% | "
                        f"erro ex = {resultado['Erro balanço exergético (%)']:.6f}% | OK"
                    )

                except Exception as erro:

                    mensagem = str(erro)

                    if "Recuperador ORC inviável" in mensagem:
                        status = "Recuperação inviável"
                        recuperacoes_inviaveis += 1

                    elif (
                        "temperatura de evaporação" in mensagem
                        or "temperatura crítica" in mensagem
                        or "pinch" in mensagem.lower()
                    ):
                        status = "Sem solução física"
                        casos_sem_solucao_fisica += 1

                    else:
                        status = "Sem solução"
                        casos_sem_solucao += 1

                    resultados.append({
                        "Fluido": fluido,
                        "Configuração ORC": configuracao_orc,
                        "epsilon Brayton": float(epsilon_b),
                        "epsilon recuperador ORC": float(epsilon_orc),
                        "Perda de pressão (%)": perda_pressao * 100.0,
                        "Status": status,
                        "Mensagem": mensagem
                    })

                    print(
                        f"[{contador}/{total_casos}] "
                        f"{fluido} | perda = {perda_pressao * 100:.0f}% | "
                        f"epsilon Brayton = {epsilon_b:.1f} | "
                        f"epsilon ORC = {epsilon_orc:.1f} | {status}: {mensagem}"
                    )

# 27. ORGANIZAÇÃO DOS RESULTADOS E VERIFICAÇÃO DA 2ª LEI

# Tolerância pequena apenas para evitar que ruídos numéricos muito próximos
# de zero sejam classificados como violação física.
# Valores negativos maiores que essa tolerância tornam o caso não admissível.
TOL_ED_NEGATIVA_KW = 1e-6
TOL_FECHAMENTO_EXERGIA = 0.1

df_resultados = pd.DataFrame(resultados)

df_solucoes = df_resultados[
    df_resultados["Status"] == "Solução encontrada"
].copy()

# Identifica automaticamente as colunas de destruição de exergia
# dos componentes. Não entram aqui o balanço global nem a coluna
# específica repetida do evaporador.

colunas_ed_componentes = [
    coluna for coluna in df_solucoes.columns
    if coluna.startswith("E_D ")
    and coluna.endswith("(kW)")
    and coluna not in [
        "E_D global (kW)",
        "E_D evaporador VCC (kW)"
    ]
]

# Verificação física pela Segunda Lei

if len(df_solucoes) > 0:

    admissibilidade_exergia = []
    componentes_negativos = []
    menor_ed_componente = []

    for _, linha in df_solucoes.iterrows():

        negativos = []
        valores_ed = []

        for coluna in colunas_ed_componentes:

            valor = linha[coluna]

            if pd.notna(valor):
                valores_ed.append(float(valor))

                if valor < -TOL_ED_NEGATIVA_KW:
                    nome_componente = coluna.replace("E_D ", "").replace(" (kW)", "")
                    negativos.append(nome_componente)

        if len(negativos) == 0:
            admissibilidade_exergia.append("Sim")
            componentes_negativos.append("")
        else:
            admissibilidade_exergia.append("Não")
            componentes_negativos.append("; ".join(negativos))

        if len(valores_ed) > 0:
            menor_ed_componente.append(min(valores_ed))
        else:
            menor_ed_componente.append(np.nan)

    df_solucoes["Admissibilidade exergética?"] = admissibilidade_exergia
    df_solucoes["Componente com E_D negativa"] = componentes_negativos
    df_solucoes["Menor E_D de componente (kW)"] = menor_ed_componente

    df_solucoes["Fechamento exergético OK?"] = np.where(
        df_solucoes["Erro balanço exergético (%)"] < TOL_FECHAMENTO_EXERGIA,
        "Sim",
        "Não"
    )

    # Critério operacional dos gases de exaustão.
    df_criterio_180 = df_solucoes[
        df_solucoes["Limite 180 °C atendido?"] == "Sim"
    ].copy()

    # Base final utilizada nas análises do TCC:
    # 1) solução convergida;
    # 2) gases de exaustão >= 180 °C;
    # 3) nenhuma destruição de exergia negativa nos componentes;
    # 4) potência positiva da turbina ORC e vazão positiva do VCC.
    df_casos_admissiveis = df_solucoes[
        (df_solucoes["Limite 180 °C atendido?"] == "Sim")
        & (df_solucoes["Admissibilidade exergética?"] == "Sim")
        & (df_solucoes["W turbina ORC (kW)"] > 0)
        & (df_solucoes["m VCC (kg/s)"] > 0)
    ].copy()

else:

    df_criterio_180 = pd.DataFrame()
    df_casos_admissiveis = pd.DataFrame()

# informações apareçam na primeira aba do Excel quando houver solução.
if len(df_solucoes) > 0:

    colunas_novas = [
        "Admissibilidade exergética?",
        "Componente com E_D negativa",
        "Menor E_D de componente (kW)",
        "Fechamento exergético OK?"
    ]

    for coluna in colunas_novas:
        if coluna not in df_resultados.columns:
            df_resultados[coluna] = np.nan

    for indice in df_solucoes.index:
        for coluna in colunas_novas:
            df_resultados.loc[indice, coluna] = df_solucoes.loc[indice, coluna]

# 28. RESUMO DA VARREDURA

print("\n" + "=" * 100)
print("VARREDURA FINALIZADA")
print("=" * 100)
print(f"Combinações avaliadas                    = {total_casos}")
print(f"Soluções encontradas                     = {solucoes_encontradas}")
print(f"Recuperações inviáveis                   = {recuperacoes_inviaveis}")
print(f"Casos sem solução física                 = {casos_sem_solucao_fisica}")
print(f"Casos sem solução                        = {casos_sem_solucao}")

if len(df_solucoes) > 0:

    casos_180 = int((df_solucoes["Limite 180 °C atendido?"] == "Sim").sum())
    casos_exergia_ok = int((df_solucoes["Admissibilidade exergética?"] == "Sim").sum())
    casos_exergia_nao = int((df_solucoes["Admissibilidade exergética?"] == "Não").sum())
    casos_fechamento_ok = int((df_solucoes["Fechamento exergético OK?"] == "Sim").sum())
    casos_finais = len(df_casos_admissiveis)

    print(f"Casos que atendem 180 °C                 = {casos_180}")
    print(f"Casos admissíveis pela 2ª Lei            = {casos_exergia_ok}")
    print(f"Casos com E_D negativa                   = {casos_exergia_nao}")
    print(f"Fechamentos exergéticos < 0,1%           = {casos_fechamento_ok}")
    print(f"Casos finais: 180 °C + 2ª Lei            = {casos_finais}")
    print(f"Maior erro exergético (%)                = {df_solucoes['Erro balanço exergético (%)'].max():.8f}")
    print(f"Maior erro evaporador (%)                = {df_solucoes['Erro balanço evaporador (%)'].max():.8f}")
    print(f"Menor pinch GV (K)                       = {df_solucoes['Pinch final GV (K)'].min():.6f}")
    print(f"Menor pinch gases-CO2 (K)                = {df_solucoes['Pinch final gases-CO2 (K)'].min():.6f}")
    print(f"Menor pinch evaporador (K)               = {df_solucoes['Pinch final evaporador VCC (K)'].min():.6f}")
    print(f"Menor E_D de componente (kW)             = {df_solucoes['Menor E_D de componente (kW)'].min():.6f}")

    if casos_exergia_nao > 0:
        print("\nComponentes responsáveis pelos casos não admissíveis:")
        print(
            df_solucoes.loc[
                df_solucoes["Admissibilidade exergética?"] == "Não",
                "Componente com E_D negativa"
            ].value_counts().to_string()
        )

print("=" * 100)

# 29. TABELAS-RESUMO PARA O EXCEL

if len(df_solucoes) > 0:

    resumo_fluidos = (
        df_solucoes
        .groupby("Fluido")
        .agg(
            Casos=("Status", "size"),
            Eta_ex_min=("Eficiência exergética global (%)", "min"),
            Eta_ex_max=("Eficiência exergética global (%)", "max"),
            ED_min=("E_D global (kW)", "min"),
            ED_max=("E_D global (kW)", "max"),
            EP_frio_min=("E produto frigorífico (kW)", "min"),
            EP_frio_max=("E produto frigorífico (kW)", "max"),
            Erro_ex_max=("Erro balanço exergético (%)", "max")
        )
        .reset_index()
    )

    # Resumo apenas da base final que será usada nos gráficos e discussões.
    if len(df_casos_admissiveis) > 0:

        resumo_admissiveis = (
            df_casos_admissiveis
            .groupby("Fluido")
            .agg(
                Casos_admissiveis=("Status", "size"),
                Eta_ex_min=("Eficiência exergética global (%)", "min"),
                Eta_ex_media=("Eficiência exergética global (%)", "mean"),
                Eta_ex_max=("Eficiência exergética global (%)", "max"),
                ED_global_min=("E_D global (kW)", "min"),
                ED_global_media=("E_D global (kW)", "mean"),
                ED_global_max=("E_D global (kW)", "max"),
                T_gases_min=("T gases saída (°C)", "min"),
                T_gases_max=("T gases saída (°C)", "max"),
                Erro_ex_max=("Erro balanço exergético (%)", "max")
            )
            .reset_index()
        )

    else:
        resumo_admissiveis = pd.DataFrame()

    resumo_componentes = []

    for coluna in colunas_ed_componentes:

        nome = coluna.replace("E_D ", "").replace(" (kW)", "")

        resumo_componentes.append({
            "Componente": nome,
            "E_D mínimo (kW) - todos": df_solucoes[coluna].min(),
            "E_D médio (kW) - todos": df_solucoes[coluna].mean(),
            "E_D máximo (kW) - todos": df_solucoes[coluna].max(),
            "Casos com E_D negativa": int((df_solucoes[coluna] < -TOL_ED_NEGATIVA_KW).sum()),
            "E_D mínimo (kW) - admissíveis": (
                df_casos_admissiveis[coluna].min()
                if len(df_casos_admissiveis) > 0 else np.nan
            ),
            "E_D médio (kW) - admissíveis": (
                df_casos_admissiveis[coluna].mean()
                if len(df_casos_admissiveis) > 0 else np.nan
            ),
            "E_D máximo (kW) - admissíveis": (
                df_casos_admissiveis[coluna].max()
                if len(df_casos_admissiveis) > 0 else np.nan
            )
        })

    resumo_componentes = pd.DataFrame(resumo_componentes)

    # Tabela específica dos casos rejeitados pela Segunda Lei.
    df_rejeitados_exergia = df_solucoes[
        df_solucoes["Admissibilidade exergética?"] == "Não"
    ].copy()

else:

    resumo_fluidos = pd.DataFrame()
    resumo_admissiveis = pd.DataFrame()
    resumo_componentes = pd.DataFrame()
    df_rejeitados_exergia = pd.DataFrame()

# 30. EXPORTAÇÃO FINAL

nome_excel = "resultados_integrado_turbina_ORC_VCC.xlsx"
nome_csv = "resultados_integrado_turbina_ORC_VCC.csv"

with pd.ExcelWriter(nome_excel, engine="openpyxl") as writer:

    df_resultados.to_excel(
        writer,
        sheet_name="Todos os casos",
        index=False
    )

    df_solucoes.to_excel(
        writer,
        sheet_name="Soluções encontradas",
        index=False
    )

    df_criterio_180.to_excel(
        writer,
        sheet_name="Critério 180C",
        index=False
    )

    df_casos_admissiveis.to_excel(
        writer,
        sheet_name="Casos admissíveis",
        index=False
    )

    df_rejeitados_exergia.to_excel(
        writer,
        sheet_name="Rejeitados 2a Lei",
        index=False
    )

    resumo_fluidos.to_excel(
        writer,
        sheet_name="Resumo exergético",
        index=False
    )

    resumo_admissiveis.to_excel(
        writer,
        sheet_name="Resumo admissíveis",
        index=False
    )

    resumo_componentes.to_excel(
        writer,
        sheet_name="Destruição componentes",
        index=False
    )

# O CSV final contém somente os casos que passaram pelos dois filtros físicos,
# pois essa é a base que será usada nas análises e nos gráficos do TCC.
df_casos_admissiveis.to_csv(
    nome_csv,
    index=False
)

print("\nArquivos gerados com sucesso:")
print(nome_excel)
print(nome_csv)

print("\nResumo da seleção final:")
print(f"Soluções convergidas                = {len(df_solucoes)}")
print(f"Atendem ao critério de 180 °C       = {len(df_criterio_180)}")
print(f"Atendem a 180 °C e à 2ª Lei         = {len(df_casos_admissiveis)}")
print(f"Rejeitados por E_D negativa          = {len(df_rejeitados_exergia)}")

print("\nAmostra dos casos finais admissíveis:")
display(df_casos_admissiveis.head(20))

# 31. DOWNLOAD AUTOMÁTICO NO GOOGLE COLAB

# Esta parte não interfere na simulação. Ela serve apenas para iniciar
# automaticamente o download do Excel quando o código terminar no Colab.
try:
    from google.colab import files
    files.download(nome_excel)
except Exception:
    print("\nDownload automático não iniciado.")
    print("O arquivo foi salvo no ambiente com o nome:")
    print(nome_excel)
