import io
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ============================================================
# NEUROVOICE 2.0 — laboratório educativo de análise acústica
# ============================================================

st.set_page_config(
    page_title="NeuroVoice 2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 45%);}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 20px;
        background: linear-gradient(120deg, #172554 0%, #1d4ed8 60%, #0891b2 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin-bottom: .25rem;}
    .hero p {color: #e0f2fe; font-size: 1.05rem; margin-bottom: 0;}
    .small-note {color: #475569; font-size: .9rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 1rem; border-radius: 14px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 NeuroVoice 2.0</h1>
  <p>Laboratório interativo de análise acústica da fala</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Explore como diferentes tarefas de fala produzem sinais acústicos distintos. "
    "O sistema mede características do áudio e ajuda a formular perguntas de neurociência."
)

with st.sidebar:
    st.header("🧪 Protocolo experimental")
    participante = st.text_input(
        "Código anônimo do participante",
        placeholder="Ex.: P01",
        help="Não digite seu nome ou informações pessoais identificáveis."
    )
    st.caption("Use um código anônimo. Evite guardar ou compartilhar gravações sem consentimento.")
    st.divider()
    st.subheader("Condições recomendadas")
    st.markdown(
        "- Ambiente silencioso\n"
        "- Microfone a distância semelhante\n"
        "- Volume de voz confortável\n"
        "- Mesma frase para todos\n"
        "- Pausas entre as tarefas"
    )
    st.divider()
    st.caption("Protótipo educativo; não é dispositivo médico.")

TAREFAS = {
    "Vogal sustentada": {
        "instrucao": "Inspire normalmente e diga “aaaa” em tom confortável por cerca de 4–6 segundos. Não force a voz.",
        "objetivo": "Observar características da fonação sustentada.",
        "prompt": "Diga: aaaaaa…",
    },
    "Articulação": {
        "instrucao": "Repita “pa-ta-ka” em ritmo confortável por aproximadamente 5 segundos. Não acelere além do que for natural.",
        "objetivo": "Observar aspectos temporais do padrão articulatório.",
        "prompt": "Repita: pa-ta-ka, pa-ta-ka…",
    },
    "Leitura padronizada": {
        "instrucao": "Leia a frase abaixo em voz natural, sem tentar imitar uma voz específica.",
        "objetivo": "Observar características acústicas durante a leitura.",
        "prompt": "“Hoje o tempo está agradável e vou ler esta frase com a minha voz natural.”",
    },
    "Fala espontânea": {
        "instrucao": "Descreva uma imagem neutra ou fale sobre um lugar que você conhece por cerca de 15–20 segundos. Não compartilhe dados pessoais.",
        "objetivo": "Observar ritmo, pausas e variação da fala espontânea.",
        "prompt": "Tema: descreva um parque, uma paisagem ou uma cena cotidiana.",
    },
}

if "resultados" not in st.session_state:
    st.session_state.resultados = {}

abas = st.tabs([
    "🧪 Experimento",
    "🎙️ Gravação e análise",
    "📊 Comparação",
    "🧠 Neurociência",
    "🔬 Método e limites",
])

with abas[0]:
    st.subheader("Desenho do experimento")
    st.write(
        "A pergunta principal é: **como as características acústicas variam entre diferentes tarefas de fala?**"
    )
    st.markdown("### Etapas")
    st.markdown(
        "1. Escolha uma tarefa de fala.\n"
        "2. Leia as instruções e grave em condições confortáveis.\n"
        "3. O programa calcula medidas acústicas e apresenta gráficos.\n"
        "4. Repita com outras tarefas e compare os resultados."
    )
    st.info(
        "Hipótese de trabalho: tarefas diferentes podem produzir padrões acústicos diferentes, "
        "porque variam em fonação, articulação, prosódia e planejamento da fala."
    )
    st.markdown("### Tarefas disponíveis")
    for nome, info in TAREFAS.items():
        with st.expander(nome):
            st.write(info["instrucao"])
            st.caption(f"Objetivo: {info['objetivo']}")
            st.markdown(f"**Enunciado:** {info['prompt']}")

with abas[1]:
    st.subheader("Grave uma tarefa")
    tarefa = st.selectbox("Escolha a tarefa", list(TAREFAS.keys()))
    st.info(TAREFAS[tarefa]["instrucao"])
    st.markdown(f"**Enunciado:** {TAREFAS[tarefa]['prompt']}")
    st.caption("Dica: mantenha o microfone a uma distância semelhante em todas as gravações.")
    audio = st.audio_input("Gravar áudio")

    if audio is not None:
        try:
            buffer_audio = io.BytesIO(audio.getvalue())
            sinal, sr = sf.read(buffer_audio, dtype="float32")

            if sinal.ndim == 2:
                sinal = np.mean(sinal, axis=1)

            sinal = np.asarray(sinal, dtype=np.float32)
            sinal = np.nan_to_num(sinal)
            duracao = len(sinal) / sr if sr else 0

            if duracao < 1.0:
                st.error("A gravação é muito curta. Grave pelo menos 1 segundo e tente novamente.")
            elif sr < 1000:
                st.error("A taxa de amostragem é muito baixa para esta análise.")
            elif np.max(np.abs(sinal)) < 1e-4:
                st.error("O áudio parece estar quase silencioso. Confira o microfone e grave novamente.")
            else:
                # Remover componente DC para uma análise mais estável
                sinal = sinal - np.mean(sinal)

                with st.spinner("Analisando as características acústicas..."):
                    hop = 512
                    f0 = librosa.yin(
                        sinal, fmin=65, fmax=500, sr=sr,
                        frame_length=2048, hop_length=hop
                    )
                    # Estimativa aproximada de quadros sonoros usando energia RMS
                    rms_frames = librosa.feature.rms(
                        y=sinal, frame_length=2048, hop_length=hop
                    )[0]
                    f0 = f0[:len(rms_frames)]
                    rms_frames = rms_frames[:len(f0)]
                    f0_valida = f0[
                        np.isfinite(f0) & (f0 >= 65) & (f0 <= 500) & (rms_frames > 0.005)
                    ]

                    rms_global = float(np.sqrt(np.mean(sinal ** 2)))
                    pico = float(np.max(np.abs(sinal)))
                    centroide_frames = librosa.feature.spectral_centroid(
                        y=sinal, sr=sr, n_fft=2048, hop_length=hop
                    )[0]
                    centroide_mediano = float(np.nanmedian(centroide_frames))

                    # Pausas aproximadas a partir de energia relativa; não é detector clínico.
                    limiar = max(float(np.max(rms_frames)) * 0.12, 0.003)
                    ativo = rms_frames > limiar
                    proporcao_pausas = float(1 - np.mean(ativo)) if len(ativo) else np.nan

                    # Taxa de fala estimada por sílabas não é inferida automaticamente.
                    # Em vez disso, mostramos duração e proporção de pausas como medidas robustas.
                    f0_mediana = float(np.median(f0_valida)) if len(f0_valida) else np.nan
                    f0_dp = float(np.std(f0_valida)) if len(f0_valida) else np.nan

                    # Qualidade de gravação aproximada: clipping digital
                    clipping = float(np.mean(np.abs(sinal) >= 0.99))

                st.success("Análise concluída.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Duração", f"{duracao:.2f} s")
                c2.metric("F0 mediana", f"{f0_mediana:.1f} Hz" if np.isfinite(f0_mediana) else "Indisponível")
                c3.metric("RMS", f"{rms_global:.4f}")
                c4.metric("Pausas estimadas", f"{proporcao_pausas * 100:.1f}%" if np.isfinite(proporcao_pausas) else "Indisponível")

                st.divider()
                st.subheader("Visualização do sinal")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Forma de onda**")
                    tempo = np.arange(len(sinal)) / sr
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.plot(tempo, sinal, linewidth=0.7)
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Amplitude relativa")
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with col_b:
                    st.markdown("**Frequência fundamental estimada (F0)**")
                    tempo_f0 = librosa.frames_to_time(
                        np.arange(len(f0)), sr=sr, hop_length=hop
                    )
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.plot(tempo_f0, f0, linewidth=0.8)
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Frequência (Hz)")
                    ax.set_ylim(50, 520)
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.markdown("**Espectrograma**")
                D = librosa.stft(sinal, n_fft=2048, hop_length=hop)
                D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                img = librosa.display.specshow(
                    D_db, sr=sr, hop_length=hop, x_axis="time",
                    y_axis="hz", ax=ax
                )
                ax.set_ylim(0, min(5000, sr / 2))
                ax.set_title("Energia do sinal em diferentes frequências")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Resumo das medidas")
                dados = [
                    ("Duração", f"{duracao:.3f} s", "Tempo total da gravação"),
                    ("Frequência fundamental mediana", f"{f0_mediana:.2f} Hz" if np.isfinite(f0_mediana) else "Indisponível", "Estimativa da frequência de vibração vocal em quadros selecionados"),
                    ("Variabilidade da F0", f"{f0_dp:.2f} Hz" if np.isfinite(f0_dp) else "Indisponível", "Desvio-padrão das estimativas de F0; não equivale ao jitter clínico"),
                    ("Amplitude RMS", f"{rms_global:.5f}", "Amplitude digital média quadrática; não é intensidade sonora calibrada"),
                    ("Centroide espectral mediano", f"{centroide_mediano:.2f} Hz", "Resumo da distribuição de energia por frequência"),
                    ("Pausas estimadas", f"{proporcao_pausas * 100:.1f}%" if np.isfinite(proporcao_pausas) else "Indisponível", "Aproximação baseada em limiar de energia"),
                    ("Possível clipping", f"{clipping * 100:.3f}%", "Proporção de amostras próximas ao limite digital"),
                ]
                df = pd.DataFrame(dados, columns=["Medida", "Resultado", "Interpretação"])
                st.dataframe(df, use_container_width=True, hide_index=True)

                if clipping > 0.01:
                    st.warning("Há sinais de possível clipping. Tente gravar com o microfone um pouco mais distante.")
                if proporcao_pausas > 0.7 and tarefa != "Fala espontânea":
                    st.warning("Muitas pausas foram estimadas. Verifique se a gravação corresponde à tarefa escolhida.")

                st.caption(
                    "As estimativas dependem da qualidade do áudio, do microfone, do ambiente, "
                    "da tarefa e das características individuais."
                )

                # Armazenar apenas métricas para a comparação dentro da sessão.
                # O áudio bruto não é guardado no estado da aplicação.
                chave = f"{participante.strip() or 'P'} — {tarefa}"
                st.session_state.resultados[chave] = {
                    "Participante": participante.strip() or "P",
                    "Tarefa": tarefa,
                    "Duração (s)": duracao,
                    "F0 mediana (Hz)": f0_mediana,
                    "Variabilidade F0 (Hz)": f0_dp,
                    "RMS": rms_global,
                    "Centroide mediano (Hz)": centroide_mediano,
                    "Pausas estimadas (%)": proporcao_pausas * 100,
                }
                st.success("As medidas desta tarefa foram adicionadas à comparação da sessão.")
        except Exception as erro:
            st.error("Não foi possível analisar esta gravação. Tente novamente com um áudio em formato compatível.")
            st.caption(f"Detalhe técnico: {type(erro).__name__}")

with abas[2]:
    st.subheader("Comparação entre tarefas")
    st.write(
        "Grave duas ou mais tarefas na aba **Gravação e análise**. "
        "A tabela abaixo reúne apenas as medidas numéricas necessárias à comparação durante esta sessão."
    )
    if st.session_state.resultados:
        df_comp = pd.DataFrame(list(st.session_state.resultados.values()))
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        metricas_disponiveis = [
            "Duração (s)", "F0 mediana (Hz)", "Variabilidade F0 (Hz)",
            "RMS", "Centroide mediano (Hz)", "Pausas estimadas (%)"
        ]
        metrica = st.selectbox("Escolha uma medida para visualizar", metricas_disponiveis)
        df_plot = df_comp[["Tarefa", metrica]].copy()
        df_plot[metrica] = pd.to_numeric(df_plot[metrica], errors="coerce")
        df_plot = df_plot.dropna()

        if not df_plot.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(df_plot["Tarefa"], df_plot[metrica])
            ax.set_ylabel(metrica)
            ax.set_xlabel("Tarefa de fala")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Não há valores disponíveis para esta medida.")
        st.caption(
            "As diferenças são descritivas. Não indicam, isoladamente, doença, estado emocional "
            "ou alterações de neurotransmissores."
        )
        if st.button("Limpar comparação desta sessão"):
            st.session_state.resultados = {}
            st.rerun()
    else:
        st.info("Ainda não há medidas nesta sessão. Grave uma tarefa para começar a comparação.")

with abas[3]:
    st.subheader("Da acústica à neurociência")
    st.markdown("### Como o cérebro participa da fala?")
    st.write(
        "A fala depende da interação entre planejamento linguístico, controle motor, feedback auditivo "
        "e coordenação dos músculos respiratórios, laríngeos e articulatórios."
    )
    with st.expander("Gânglios da base e controle motor"):
        st.write(
            "Os gânglios da base participam da seleção e regulação de ações motoras. Pesquisas investigam "
            "como alterações nesses circuitos podem estar associadas a mudanças na voz e na fala. "
            "Um sinal acústico isolado, porém, não revela diretamente a atividade desses circuitos."
        )
    with st.expander("Dopamina e neuromodulação"):
        st.write(
            "A dopamina atua em diferentes circuitos cerebrais e receptores, influenciando funções "
            "motoras e outras funções. O NeuroVoice não mede dopamina, não identifica receptores ativos "
            "e não consegue deduzir concentrações de neurotransmissores pela gravação."
        )
    with st.expander("O que as medidas acústicas descrevem?"):
        st.markdown(
            "- **F0:** estimativa da frequência fundamental vocal.\n"
            "- **Variabilidade da F0:** dispersão das estimativas de frequência ao longo da gravação.\n"
            "- **RMS:** amplitude digital média quadrática.\n"
            "- **Centroide espectral:** característica resumida da distribuição de energia no espectro.\n"
            "- **Pausas estimadas:** proporção de quadros com energia abaixo de um limiar definido."
        )
    st.info(
        "Conexão científica: características acústicas podem ser estudadas como possíveis biomarcadores digitais, "
        "mas sua utilidade clínica exige estudos controlados, amostras adequadas e validação independente."
    )

with abas[4]:
    st.subheader("Método, validade e limitações")
    st.markdown("### Pergunta de pesquisa")
    st.write("Como as características acústicas extraídas por um programa variam entre diferentes tarefas de fala?")
    st.markdown("### Hipótese")
    st.write(
        "As medidas acústicas podem variar entre tarefas, pois uma vogal sustentada, a articulação repetida, "
        "a leitura e a fala espontânea impõem exigências diferentes à produção da fala."
    )
    st.markdown("### Variáveis")
    st.markdown(
        "- **Variável independente:** tipo de tarefa de fala.\n"
        "- **Variáveis dependentes:** duração, F0 estimada, variabilidade da F0, RMS, centroide espectral e pausas aproximadas.\n"
        "- **Fatores de confusão:** ruído, distância do microfone, volume, fadiga, idioma, idade, características vocais e dispositivo."
    )
    st.markdown("### Limitações importantes")
    st.markdown(
        "- O algoritmo de F0 pode errar em áudio ruidoso ou em segmentos não sonoros.\n"
        "- A amplitude digital depende do microfone e da distância; não equivale a decibéis calibrados.\n"
        "- A proporção de pausas é uma aproximação por energia, não uma medida clínica validada.\n"
        "- Uma comparação pequena em sala de aula não permite generalizar para uma população.\n"
        "- O sistema não diagnostica Parkinson ou qualquer outra condição e não mede hormônios, neurotransmissores ou receptores."
    )
    st.warning(
        "É um projeto educativo de processamento de sinais. Para coleta sistemática com participantes, "
        "obtenha autorização da professora responsável, explique a atividade e recolha consentimento. "
        "Não publique vozes ou dados identificáveis."
    )

st.divider()
st.caption("NEUROVOICE 2.0 • Protótipo educativo de análise acústica • Sem finalidade diagnóstica • Sem exportação de arquivos")
