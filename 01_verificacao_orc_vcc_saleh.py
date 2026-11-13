import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI
from tespy.networks import Network
from tespy.components import CycleCloser, Valve, Compressor, Pump, SimpleHeatExchanger, Turbine, Merge, Splitter
from tespy.connections import Connection

# Fluidos de trabalho e resultados de referência de Saleh (2018), Tabela 3.
fluidos = {'R600': 'R600', 'R600a': 'R600a', 'R1234ze(E)': 'R1234ze(E)', 'R602': 'n-Hexane'}
saleh = {
    'R600': {'Q_eva': 301.1, 'W_turb': 59.59, 'eta_ORC': 11.67, 'COP_VCC': 5.05, 'COP_S': 0.589, 'm_total_x100': 0.720},
    'R600a': {'Q_eva': 270.7, 'W_turb': 54.56, 'eta_ORC': 11.56, 'COP_VCC': 4.96, 'COP_S': 0.574, 'm_total_x100': 0.813},
    'R1234ze(E)': {'Q_eva': 136.3, 'W_turb': 28.17, 'eta_ORC': 11.26, 'COP_VCC': 4.84, 'COP_S': 0.545, 'm_total_x100': 1.763},
    'R602': {'Q_eva': 303.7, 'W_turb': 58.93, 'eta_ORC': 11.56, 'COP_VCC': 5.16, 'COP_S': 0.596, 'm_total_x100': 0.687}
}

def diferenca_relativa(valor_codigo, valor_saleh):
    return abs((valor_codigo - valor_saleh) / valor_saleh) * 100

def simular(nome_fluido, fluido):
    # Criação dos componentes e das conexões do ciclo ORC--VCC.
    nw = Network()
    nw.units.set_defaults(temperature='degC', pressure='MPa', enthalpy='kJ/kg', power='kW', heat='kW')
    sp = Splitter('separador')
    cc = CycleCloser('fechamento')
    ev = SimpleHeatExchanger('evaporador')
    cp = Compressor('compressor')
    co = SimpleHeatExchanger('condensador')
    ve = Valve('valvula')
    bo = Pump('bomba')
    aq = SimpleHeatExchanger('aquecedor')
    tu = Turbine('turbina')
    cm = Merge('mistura')
    c1 = Connection(cc, 'out1', co, 'in1', label='1')
    c2 = Connection(co, 'out1', sp, 'in1', label='2')
    c3 = Connection(sp, 'out1', bo, 'in1', label='3')
    c4 = Connection(bo, 'out1', aq, 'in1', label='4')
    c5 = Connection(aq, 'out1', tu, 'in1', label='5')
    c6 = Connection(tu, 'out1', cm, 'in1', label='6')
    c7 = Connection(sp, 'out2', ve, 'in1', label='7')
    c8 = Connection(ve, 'out1', ev, 'in1', label='8')
    c9 = Connection(ev, 'out1', cp, 'in1', label='9')
    c10 = Connection(cp, 'out1', cm, 'in2', label='10')
    c0 = Connection(cm, 'out1', cc, 'in1', label='0')
    nw.add_conns(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c0)

    # Condições de operação utilizadas na verificação.
    tu.set_attr(eta_s=0.80)
    bo.set_attr(eta_s=0.80)
    cp.set_attr(eta_s=0.75)
    ev.set_attr(pr=1.0)
    co.set_attr(pr=1.0)
    aq.set_attr(pr=1.0)
    c2.set_attr(T=35, x=0, fluid={fluido: 1})
    c9.set_attr(T=0, x=1, m=1)
    c5.set_attr(T=100, x=1)
    p_alta = PropsSI('P', 'T', 373.15, 'Q', 1, fluido) / 1e6
    p_baixa = PropsSI('P', 'T', 273.15, 'Q', 1, fluido) / 1e6
    p_cond = PropsSI('P', 'T', 308.15, 'Q', 0, fluido) / 1e6
    c5.set_attr(p0=p_alta)
    c9.set_attr(p0=p_baixa)
    c2.set_attr(p0=p_cond)

    # A vazão do ORC é ajustada até equilibrar a turbina e o compressor.
    def balanco_potencia(m_orc):
        c3.set_attr(m=m_orc)
        try:
            nw.solve(mode='design')
            return tu.P.val + cp.P.val
        except Exception:
            return np.nan

    massas = np.logspace(-3, 2, 100)
    valores = [balanco_potencia(m) for m in massas]
    intervalo = None
    for i in range(len(massas) - 1):
        if np.isfinite(valores[i]) and np.isfinite(valores[i + 1]) and valores[i] * valores[i + 1] < 0:
            intervalo = (massas[i], massas[i + 1])
            break
    if intervalo is None:
        raise RuntimeError(f'Não foi encontrado equilíbrio de potência para {nome_fluido}.')
    m_orc = brentq(balanco_potencia, intervalo[0], intervalo[1])
    c3.set_attr(m=m_orc)
    nw.solve(mode='design')

    # Cálculo dos indicadores para comparação com a referência.
    m_vcc = c9.m.val
    W_turb = abs(tu.P.val)
    W_comp = abs(cp.P.val)
    W_pump = abs(bo.P.val)
    Q_eva = abs(ev.Q.val)
    Q_boiler = abs(aq.Q.val)
    COP_VCC = Q_eva / W_comp
    eta_ORC = W_turb / (Q_boiler + W_pump) * 100
    COP_S = Q_eva / (Q_boiler + W_pump)
    m_total_x100 = ((m_orc + m_vcc) / Q_eva) * 100
    return {'Fluido': nome_fluido, 'm_ORC': m_orc, 'Q_eva': Q_eva, 'W_turb': W_turb,
            'eta_ORC': eta_ORC, 'COP_VCC': COP_VCC, 'COP_S': COP_S, 'm_total_x100': m_total_x100}

resultados = []
for nome, fluido in fluidos.items():
    print(f'\nExecutando {nome}...')
    try:
        resultados.append(simular(nome, fluido))
        print('Convergência obtida.')
    except Exception as erro:
        print(f'Falha para {nome}: {erro}')

df_codigo = pd.DataFrame(resultados)
parametros = {'Q_eva': 'Q_eva [kW]', 'W_turb': 'W_turb [kW]', 'eta_ORC': 'eta_ORC [%]',
              'COP_VCC': 'COP_VCC [-]', 'COP_S': 'COP_S [-]', 'm_total_x100': 'm_total x100 [kg/s/kW]'}
comparacao = []
for _, resultado in df_codigo.iterrows():
    fluido = resultado['Fluido']
    for chave, parametro in parametros.items():
        codigo = resultado[chave]
        referencia = saleh[fluido][chave]
        comparacao.append({'Fluido': fluido, 'Parâmetro': parametro, 'Código': codigo,
                           'Saleh': referencia, 'Diferença relativa [%]': diferenca_relativa(codigo, referencia)})
df_verificacao = pd.DataFrame(comparacao)
print('\nVERIFICAÇÃO DO MODELO ORC-VCC COM SALEH (2018)\n')
print(df_verificacao.round(4).to_string(index=False))

# Apresentação da tabela de comparação.
tabela_plot = df_verificacao.copy()
for coluna in ['Código', 'Saleh']:
    tabela_plot[coluna] = tabela_plot[coluna].map(lambda x: f'{x:.4f}')
tabela_plot['Diferença relativa [%]'] = tabela_plot['Diferença relativa [%]'].map(lambda x: f'{x:.2f}')
fig, ax = plt.subplots(figsize=(13, 10))
ax.axis('off')
tabela = ax.table(cellText=tabela_plot.values, colLabels=tabela_plot.columns,
                  cellLoc='center', colLoc='center', loc='center')
tabela.auto_set_font_size(False)
tabela.set_fontsize(9)
tabela.scale(1, 1.4)
plt.title('Verificação do modelo ORC-VCC com Saleh (2018)', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

resumo = df_verificacao.groupby('Fluido')['Diferença relativa [%]'].max().reset_index()
resumo.columns = ['Fluido', 'Maior diferença relativa [%]']
print('\nMAIOR DIFERENÇA RELATIVA PARA CADA FLUIDO\n')
print(resumo.round(3).to_string(index=False))
