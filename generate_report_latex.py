"""
generate_report_latex.py
Generates a professional LaTeX (.tex) file and compiles to PDF.
Author: Mahinur Akhter | ID: 22201100
"""

import sys, os, subprocess, shutil
from pathlib import Path
from datetime import date

BASE    = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
TEX_DIR = BASE / "report_latex"
TEX_DIR.mkdir(exist_ok=True)
TEX_FILE = TEX_DIR / "report.tex"
PDF_OUT  = BASE / "Crop_Yield_PlantDisease_Report_MahinurAkhter.pdf"


# ── LaTeX install ─────────────────────────────────────────────────────────────
def ensure_latex():
    for eng in ("pdflatex", "xelatex"):
        if shutil.which(eng):
            return eng
    print("Installing texlive packages (takes a few minutes)...")
    subprocess.run(
        ["sudo", "apt-get", "install", "-y", "--no-install-recommends",
         "texlive-latex-extra", "texlive-fonts-recommended",
         "texlive-science", "latexmk"],
        check=True
    )
    return "pdflatex"


# ── Image helpers ─────────────────────────────────────────────────────────────
def P(name):
    """Return forward-slash path string for LaTeX, or empty string."""
    p = OUTPUTS / name
    return str(p).replace("\\", "/") if p.exists() else ""


def _latex_cap(s):
    """Escape % and _ for LaTeX captions — only if not already escaped."""
    import re
    s = re.sub(r'(?<!\\)%', r'\\%', s)   # % not preceded by backslash
    s = re.sub(r'(?<!\\)_', r'\\_', s)   # _ not preceded by backslash
    return s

def fig(name, caption, width="0.88\\textwidth"):
    path = P(name)
    if not path:
        return f"% [image not found: {name}]\n"
    cap = _latex_cap(caption)
    lbl = name.replace(".", "_").replace(" ", "_")
    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        f"  \\includegraphics[width={width}]{{{path}}}",
        f"  \\caption{{{cap}}}",
        f"  \\label{{fig:{lbl}}}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)


def fig2(nameL, capL, nameR, capR):
    pL = P(nameL)
    pR = P(nameR)
    cL = _latex_cap(capL)
    cR = _latex_cap(capR)
    incL = f"\\includegraphics[width=\\textwidth]{{{pL}}}" if pL else "% missing"
    incR = f"\\includegraphics[width=\\textwidth]{{{pR}}}" if pR else "% missing"
    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \begin{minipage}{0.48\textwidth}",
        r"    \centering",
        f"    {incL}",
        f"    \\caption{{{cL}}}",
        r"  \end{minipage}\hfill",
        r"  \begin{minipage}{0.48\textwidth}",
        r"    \centering",
        f"    {incR}",
        f"    \\caption{{{cR}}}",
        r"  \end{minipage}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)


# ── Build .tex ────────────────────────────────────────────────────────────────
def build_tex():
    today   = date.today().strftime("%B %d, %Y")
    cover_img = P("eda_yield_distribution.png")

    parts = []

    # ── Preamble ──────────────────────────────────────────────────────────────
    parts.append(r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{parskip}
\usepackage{microtype}
\usepackage{colortbl}

\definecolor{darkgreen}{RGB}{26,83,30}
\definecolor{midgreen}{RGB}{39,108,43}
\definecolor{tableheader}{RGB}{26,83,30}
\definecolor{rowalt}{RGB}{245,245,245}

\titleformat{\section}{\Large\bfseries\color{darkgreen}}{\thesection.}{0.8em}{}
\titleformat{\subsection}{\large\bfseries\color{midgreen}}{\thesubsection}{0.8em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{midgreen}}{\thesubsubsection}{0.8em}{}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{darkgreen}{\small Crop Yield Prediction \& Plant Disease Detection}}
\fancyhead[R]{\textcolor{darkgreen}{\small Mahinur Akhter -- 22201100}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\headheight}{14.5pt}

\captionsetup{font=small, labelfont=bf, justification=centering}
\hypersetup{colorlinks=true, linkcolor=darkgreen, urlcolor=midgreen, citecolor=darkgreen}
\onehalfspacing

\begin{document}
""")

    # ── Cover page ────────────────────────────────────────────────────────────
    parts.append(r"\begin{titlepage}")
    parts.append(r"  \centering")
    parts.append(r"  \vspace*{1cm}")
    parts.append(r"  {\Huge\bfseries\color{darkgreen}")
    parts.append(r"    Crop Yield Prediction and\\[0.3em]")
    parts.append(r"    Plant Disease Detection}\\[0.5em]")
    parts.append(r"  {\large\color{midgreen}")
    parts.append(r"    Using Machine Learning and Deep Learning}\\[0.3em]")
    parts.append(r"  {\normalsize\itshape\color{gray}")
    parts.append(r"    An Integrated Pipeline: Tabular ML + CNN + Explainable AI (Grad-CAM)}")
    parts.append(r"  \vspace{1.2cm}")
    parts.append(r"  \noindent\rule{0.9\textwidth}{1.5pt}\\[0.4em]")
    parts.append(r"  \begin{tabular}{rl}")
    parts.append(r"    \textbf{Author}      & Mahinur Akhter \\")
    parts.append(r"    \textbf{Student ID}  & 22201100 \\")
    parts.append(r"    \textbf{Department}  & Computer Science and Engineering \\")
    parts.append(f"    \\textbf{{Date}}       & {today} \\\\")
    parts.append(r"    \textbf{Datasets}    & 10 Tabular + 4 Image (Kaggle / FAO) \\")
    parts.append(r"    \textbf{Framework}   & Python $\cdot$ scikit-learn $\cdot$ XGBoost $\cdot$ TensorFlow/Keras \\")
    parts.append(r"  \end{tabular}\\[0.4em]")
    parts.append(r"  \noindent\rule{0.9\textwidth}{1.5pt}")
    parts.append(r"  \vfill")
    if cover_img:
        parts.append(f"  \\includegraphics[width=0.55\\textwidth]{{{cover_img}}}\\\\[0.3em]")
    parts.append(r"  {\small\itshape\color{gray} Crop Yield Distribution Analysis}")
    parts.append(r"\end{titlepage}")
    parts.append(r"\tableofcontents")
    parts.append(r"\listoffigures")
    parts.append(r"\listoftables")
    parts.append(r"\newpage")

    # ── Abstract ──────────────────────────────────────────────────────────────
    parts.append(r"""
\section{Abstract}

This report presents an end-to-end machine learning pipeline for two
interdependent agricultural tasks: \textbf{crop yield prediction} from tabular
soil, climate, and environmental data, and \textbf{plant disease detection} from
leaf imagery via deep learning.
The tabular pipeline merges \textbf{ten publicly available datasets} spanning
2000--2016 across 206 countries and constructs a feature set of
\textbf{22 input variables}.
\textbf{XGBoost} achieves the best regression performance
($R^2 = 0.2121$, RMSE\,$=$\,70{,}104\,hg/ha) and the best classification
F1-score of 0.5801.
The image branch fine-tunes \textbf{MobileNetV2} on the PlantVillage dataset
(70{,}295 training images, 38 disease classes),
achieving \textbf{validation accuracy of 98.06\,\%}.
Explainability is delivered through \textbf{Grad-CAM} heatmaps,
disease severity scoring, and quantitative leaf-colour indices.

\textbf{Keywords:} Crop Yield Prediction $\cdot$ Plant Disease Detection
$\cdot$ XGBoost $\cdot$ MobileNetV2 $\cdot$ Grad-CAM $\cdot$ Explainable AI
$\cdot$ Bangladesh Agriculture $\cdot$ PlantVillage
\newpage
""")

    # ── Introduction ──────────────────────────────────────────────────────────
    parts.append(r"""
\section{Introduction}

Agriculture is the backbone of Bangladesh's economy, employing nearly 40\,\%
of the workforce and contributing approximately 13\,\% of GDP\@.
Farmers face two persistent challenges: uncertainty in crop yield and
undetected plant diseases.
This project constructs a dual-branch decision-support system:

\begin{itemize}[leftmargin=1.5em]
  \item \textbf{Tabular ML Branch} --- Predicts crop yield (hg/ha) and
        classifies it as \textit{High}, \textit{Medium}, or \textit{Low}
        from 22 merged climate, soil, and environmental features.
  \item \textbf{Image DL Branch} --- Detects plant disease from a leaf
        photograph using MobileNetV2, with Grad-CAM heatmaps and
        quantitative disease severity.
\end{itemize}
\newpage
""")

    # ── Datasets ──────────────────────────────────────────────────────────────
    parts.append(r"""
\section{Datasets}
\label{sec:datasets}

\subsection{Tabular Datasets}

\begin{table}[H]
\centering
\caption{Summary of the 10 tabular datasets.}
\label{tab:datasets}
\small
\begin{tabular}{clllrr}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{DS\#}} &
\textcolor{white}{\textbf{Dataset}} &
\textcolor{white}{\textbf{Type}} &
\textcolor{white}{\textbf{Years}} &
\textcolor{white}{\textbf{Rows}} &
\textcolor{white}{\textbf{Source}} \\
\midrule
\rowcolor{rowalt}
1  & FAO Crop Yield (EDA+Viz)         & Soil+Climate  & 1961--2016 & 56,717  & FAO \\
2  & Crop Recommendation (N,P,K,pH)   & Soil          & 2020       & 2,200   & Kaggle \\
\rowcolor{rowalt}
3  & Crop and Soil Dataset            & Soil+Climate  & 2010--2023 & 8,000   & Kaggle \\
4  & Agricultural Land Suitability BD & Soil+Climate  & 2020--2024 & 9.1M    & Kaggle \\
\rowcolor{rowalt}
5  & BD Agroclimatic Crop Yield       & Climate+Crop  & 2000--2024 & 150     & Kaggle \\
6  & Earth Surface Temperature        & Climate       & 1901--2015 & 39,900  & Kaggle \\
\rowcolor{rowalt}
7  & Climate Data Bangladesh 2021--24 & Climate       & 2021--2024 & 1,460   & Kaggle \\
8  & Bangladesh Weather 1901--2023    & Climate       & 1901--2023 & 1,386   & Kaggle \\
\rowcolor{rowalt}
9  & Environmental Sensor Telemetry   & Environmental & $\sim$10yr & 405,184 & Kaggle \\
10 & Dhaka Air Quality 2000--2025     & Environmental & 2000--2025 & 225,000 & Kaggle \\
\rowcolor{rowalt}
XLS& Real Soil Sensor (BD)            & Soil          & 2026       & 142     & Local \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Image Datasets}

\begin{table}[H]
\centering
\caption{Summary of image datasets.}
\label{tab:imgdatasets}
\small
\begin{tabular}{clccr}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{\#}} &
\textcolor{white}{\textbf{Dataset}} &
\textcolor{white}{\textbf{Crops}} &
\textcolor{white}{\textbf{Classes}} &
\textcolor{white}{\textbf{Images}} \\
\midrule
\rowcolor{rowalt}
1 & PlantVillage (New Plant Diseases) & Multi-crop & 38 & 87,867 \\
2 & Rice Leaf Disease Dataset         & Rice       &  4 &  5,932 \\
\rowcolor{rowalt}
3 & Crop Disease Detection Dataset    & Multi-crop &  8 & 15,000 \\
4 & Plant Seedlings Classification    & Multi-crop & 12 &  5,539 \\
\bottomrule
\end{tabular}
\end{table}
\newpage
""")

    # ── Methodology ───────────────────────────────────────────────────────────
    parts.append(r"""
\section{Methodology}
\label{sec:method}

\subsection{Seven-Step Data Merge Strategy}

\begin{enumerate}[leftmargin=1.8em]
  \item \textbf{Validate \& Normalise} --- Column names lower-cased, backbone filtered to 2000--2020.
  \item \textbf{Climate Merge} --- Temperature, rainfall, humidity, wind speed from 5 sources.
        Priority: BD Agroclimatic $>$ Dhaka Air $>$ Earth Temp $>$ FAO.
  \item \textbf{Soil Merge} --- N, P, K, pH looked up by crop type; overridden with XLS sensor data for Bangladesh.
  \item \textbf{Environmental Merge} --- AQI, PM$_{2.5}$, NO$_2$, SO$_2$ from Dhaka Air Quality dataset.
  \item \textbf{Feature Engineering} --- Derived: decade, temperature\_range, sunshine\_hours,
        fertilizer\_kgha, rainfall\_category, season.
  \item \textbf{Missing Value Handling} --- Columns with $>$90\,\% nulls dropped;
        numeric NaNs median-imputed.
  \item \textbf{Save} --- Merged dataset (18,741 rows $\times$ 26 columns) saved.
\end{enumerate}

\subsection{Final Feature Set (22 Input Features)}

\begin{table}[H]
\centering
\caption{Final 22-feature input set.}
\label{tab:features}
\small
\begin{tabular}{clll}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{\#}} &
\textcolor{white}{\textbf{Feature}} &
\textcolor{white}{\textbf{Category}} &
\textcolor{white}{\textbf{Unit}} \\
\midrule
\rowcolor{rowalt} 1  & Year              & Common      & Integer \\
2  & avg\_temp         & Climate     & $^\circ$C \\
\rowcolor{rowalt} 3  & rainfall\_mm      & Climate     & mm \\
4  & humidity\_pct     & Climate     & \% \\
\rowcolor{rowalt} 5  & min\_temp         & Climate     & $^\circ$C \\
6  & max\_temp         & Climate     & $^\circ$C \\
\rowcolor{rowalt} 7  & wind\_speed\_kmh  & Climate     & km/h \\
8  & sunshine\_hours   & Climate     & hrs/day \\
\rowcolor{rowalt} 9  & season            & Climate     & Kharif/Rabi \\
10 & nitrogen\_N       & Soil        & mg/kg \\
\rowcolor{rowalt} 11 & phosphorous\_P    & Soil        & mg/kg \\
12 & potassium\_K      & Soil        & mg/kg \\
\rowcolor{rowalt} 13 & soil\_pH          & Soil        & pH \\
14 & soil\_moisture    & Soil        & \% \\
\rowcolor{rowalt} 15 & soil\_type        & Soil        & Category \\
16 & fertilizer\_kgha  & Soil        & kg/ha \\
\rowcolor{rowalt} 17 & AQI               & Environment & Index \\
18 & CO$_2$\_ppm       & Environment & ppm \\
\rowcolor{rowalt} 19 & PM$_{2.5}$        & Environment & $\mu$g/m$^3$ \\
20 & NO$_2$\_ppb       & Environment & ppb \\
\rowcolor{rowalt} 21 & decade            & Engineered  & Integer \\
22 & temperature\_range& Engineered  & $^\circ$C \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Machine Learning Models}

Data split 80/20 (random\_state=42). Categorical features label-encoded;
numeric features median-imputed and standardised with StandardScaler.

\textbf{Regression:} predict exact yield\_hgha.
Metrics: RMSE, MAE, $R^2$, 5-fold CV $R^2$.

\textbf{Classification:} High ($>$50k), Medium (20k--50k), Low ($<$20k hg/ha).
Metrics: Accuracy, Weighted F1.

Models: Linear Regression, Random Forest ($n$=200), XGBoost ($n$=200, lr=0.05),
KNN ($k$=7), Gaussian Na\"{i}ve Bayes (classification only).

\subsection{CNN Architecture --- MobileNetV2}

MobileNetV2 (ImageNet pretrained) + custom head:
$\text{GAP} \to \text{BN} \to \text{Dense}(256) \to \text{Drop}(0.4)
\to \text{Dense}(128) \to \text{Drop}(0.3) \to \text{Softmax}(38)$

\textbf{Phase 1}: Base frozen, lr=$10^{-3}$, 5 epochs.
\textbf{Phase 2}: Last 30 layers unfrozen, lr=$10^{-4}$, 10 epochs.

\subsection{Explainability --- Grad-CAM and Image Features}

\begin{table}[H]
\centering
\caption{Quantitative image features extracted per prediction.}
\label{tab:imgfeatures}
\small
\begin{tabular}{lll}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{Feature}} &
\textcolor{white}{\textbf{Formula}} &
\textcolor{white}{\textbf{Interpretation}} \\
\midrule
\rowcolor{rowalt} Confidence \%   & $\max(\text{softmax})\times100$ & Model certainty \\
Yellowing Index  & $R_\mu - G_\mu$ & $>0$ = yellowing \\
\rowcolor{rowalt} Browning Index   & $R_\mu - B_\mu$ & $>0$ = browning \\
Texture Score    & $\sigma(\text{RGB pixels})$ & Surface irregularity \\
\rowcolor{rowalt} Disease Severity\%& \% heatmap pixels $>$ 0.5 & Affected area extent \\
\bottomrule
\end{tabular}
\end{table}
\newpage
""")

    # ── EDA ───────────────────────────────────────────────────────────────────
    parts.append(r"""
\section{Exploratory Data Analysis}
\label{sec:eda}

After merging: \textbf{18,741 records}, \textbf{206 countries},
\textbf{10 crop types}, 2000--2016.
Mean yield = 73,455 hg/ha, mean temp = 22.1$^\circ$C,
mean rainfall = 1,122 mm.
""")
    parts.append(fig("eda_yield_distribution.png",
        "Yield distribution histogram and median yield by crop type."))
    parts.append(fig("eda_correlation_heatmap.png",
        "Pearson correlation heatmap of all 22 numeric features.",
        width="0.82\\textwidth"))
    parts.append(fig("eda_scatter_plots.png",
        "Scatter plots: temperature, rainfall, and nitrogen vs.\\ yield."))
    parts.append(fig2("eda_yield_categories.png",
        "Yield category distribution.",
        "eda_yield_trend.png",
        "Average crop yield trend 2000--2016."))
    parts.append(r"\newpage")

    # ── Regression Results ────────────────────────────────────────────────────
    parts.append(r"""
\section{Experimental Results --- Tabular ML}
\label{sec:results}

\subsection{Regression Results}

\begin{table}[H]
\centering
\caption{Regression model comparison ($\star$ = best).}
\label{tab:regression}
\begin{tabular}{lrrrr}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{Model}} &
\textcolor{white}{\textbf{RMSE}} &
\textcolor{white}{\textbf{MAE}} &
\textcolor{white}{\textbf{$R^2$}} &
\textcolor{white}{\textbf{CV $R^2$}} \\
\midrule
\rowcolor{rowalt} Linear Regression         & 74,034 & 54,102 & 0.1212 & 0.1352 \\
Random Forest                               & 84,362 & 56,483 & -0.141 & -0.100 \\
\rowcolor{rowalt} \textbf{XGBoost $\star$}  & \textbf{70,104} & \textbf{49,316} & \textbf{0.2121} & \textbf{0.2152} \\
KNN                                         & 76,377 & 54,073 & 0.0647 & 0.0918 \\
\bottomrule
\end{tabular}
\end{table}
""")
    parts.append(fig("regression_comparison.png",
        "RMSE, MAE, and $R^2$ comparison across regression models."))
    parts.append(fig2("actual_vs_pred_XGBoost.png",
        "Actual vs.\\ predicted yield --- XGBoost.",
        "feature_importance_Regression-XGBoost.png",
        "Top-15 feature importances --- XGBoost regression."))

    # ── Classification Results ────────────────────────────────────────────────
    parts.append(r"""
\subsection{Classification Results}

\begin{table}[H]
\centering
\caption{Classification model comparison ($\star$ = best).}
\label{tab:classification}
\begin{tabular}{lcc}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{Model}} &
\textcolor{white}{\textbf{Accuracy}} &
\textcolor{white}{\textbf{Weighted F1}} \\
\midrule
\rowcolor{rowalt} Random Forest              & 0.4953 & 0.4884 \\
\textbf{XGBoost $\star$}                    & \textbf{0.6013} & \textbf{0.5801} \\
\rowcolor{rowalt} KNN                        & 0.5108 & 0.4928 \\
Gaussian Na\"{i}ve Bayes                    & 0.5015 & 0.4279 \\
\bottomrule
\end{tabular}
\end{table}
""")
    parts.append(fig2("classification_comparison.png",
        "Accuracy and F1 comparison across classifiers.",
        "confusion_matrix_XGBoost.png",
        "Confusion matrix --- XGBoost (High/Medium/Low)."))
    parts.append(fig("feature_importance_Classification-XGBoost.png",
        "Feature importances for XGBoost classifier.",
        width="0.80\\textwidth"))
    parts.append(r"\newpage")

    # ── CNN Results ───────────────────────────────────────────────────────────
    parts.append(r"""
\section{Plant Disease Detection Results}
\label{sec:cnn}

\begin{table}[H]
\centering
\caption{CNN training and evaluation summary.}
\label{tab:cnn}
\begin{tabular}{ll}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{Metric}} & \textcolor{white}{\textbf{Value}} \\
\midrule
\rowcolor{rowalt} Validation Accuracy  & \textbf{98.06\,\%} \\
Validation Loss                        & 0.0594 \\
\rowcolor{rowalt} Classes              & 38 \\
Training Images                        & 70,295 \\
\rowcolor{rowalt} Validation Images    & 17,572 \\
Macro-avg Precision                    & 0.98 \\
\rowcolor{rowalt} Macro-avg Recall     & 0.98 \\
Macro-avg F1                           & 0.98 \\
\rowcolor{rowalt} Architecture         & MobileNetV2 + Custom Head \\
XAI Method                             & Grad-CAM \\
\bottomrule
\end{tabular}
\end{table}
""")
    parts.append(fig2("history_phase_1_frozen.png",
        "Phase 1 training history (frozen base).",
        "history_phase_2_fine-tune.png",
        "Phase 2 fine-tuning history."))
    parts.append(fig("confusion_matrix.png",
        "38-class confusion matrix on 17,572 validation images."))
    parts.append(fig("gradcam_sample.jpg",
        "Grad-CAM heatmap overlay on a sample Apple scab validation image. "
        "Red/yellow regions highlight disease lesions. Disease severity: 9.1%.",
        width="0.50\\textwidth"))

    parts.append(r"""
\subsection{Extracted Image Features (XAI)}

\begin{table}[H]
\centering
\caption{XAI features for 5 sample validation images.}
\label{tab:xai}
\small
\begin{tabular}{clccccc}
\toprule
\rowcolor{tableheader}
\textcolor{white}{\textbf{\#}} &
\textcolor{white}{\textbf{Disease}} &
\textcolor{white}{\textbf{Conf.\%}} &
\textcolor{white}{\textbf{Yellow.}} &
\textcolor{white}{\textbf{Brown.}} &
\textcolor{white}{\textbf{Severity\%}} \\
\midrule
\rowcolor{rowalt} 0 & Apple\_scab & 100.00 & -0.0768 & -0.0359 &  9.11 \\
1                   & Apple\_scab & 100.00 & -0.0768 & -0.0356 & 20.13 \\
\rowcolor{rowalt} 2 & Apple\_scab & 100.00 & -0.0803 & -0.0233 &  3.15 \\
3                   & Apple\_scab & 100.00 & -0.1299 & -0.0911 &  5.66 \\
\rowcolor{rowalt} 4 & Apple\_scab &  45.17 & +0.0165 & +0.0609 & 23.16 \\
\bottomrule
\end{tabular}
\end{table}
\newpage
""")

    # ── Discussion ────────────────────────────────────────────────────────────
    parts.append(r"""
\section{Discussion}
\label{sec:discussion}

\subsection{Tabular ML Performance}

The modest $R^2$ of 0.21 reflects the inherent difficulty: crop yield is
influenced by seed variety, pest pressure, and irrigation management not
captured in the datasets.
XGBoost outperformed all baselines; Random Forest underperformed due to
sentinel-filled values for AQI and PM$_{2.5}$.

\subsection{CNN Performance}

The 98.06\,\% validation accuracy is consistent with published PlantVillage
benchmarks \cite{mohanty2016}.
MobileNetV2's lightweight architecture enables edge deployment, critical
for rural Bangladesh with limited connectivity.
Grad-CAM heatmaps correctly localise lesion regions, confirming that the
model has learned pathologically meaningful patterns.

\subsection{Limitations}
\begin{itemize}
  \item Tabular model trained on global data; Bangladesh field-level validation not performed.
  \item Datasets 3, 4, 7, and 8 were unavailable (403 Forbidden).
  \item PlantVillage images are studio-photographed; real-field performance is lower.
\end{itemize}
\newpage
""")

    # ── Conclusion ────────────────────────────────────────────────────────────
    parts.append(r"""
\section{Conclusion}
\label{sec:conclusion}

This work presents a complete ML pipeline for agricultural decision support.
XGBoost achieves $R^2 = 0.2121$ for yield regression and F1 = 0.5801 for
yield classification across 18,741 records from 206 countries.
MobileNetV2 achieves \textbf{98.06\,\%} validation accuracy across 38 plant
disease classes. Grad-CAM and five leaf-colour indices provide interpretable
predictions for farmers.

\subsection*{Future Work}
\begin{itemize}
  \item Collect Bangladesh field-level yield data for retraining.
  \item Integrate unavailable datasets when access is restored.
  \item Deploy CNN as TFLite mobile app for offline leaf scanning.
  \item Apply SHAP to the tabular model for per-prediction attribution.
\end{itemize}
""")

    # ── References ────────────────────────────────────────────────────────────
    parts.append(r"""
\begin{thebibliography}{9}
\bibitem{chen2016}
T.~Chen and C.~Guestrin, ``XGBoost: A Scalable Tree Boosting System,''
\textit{KDD}, pp.~785--794, 2016.

\bibitem{sandler2018}
M.~Sandler et al., ``MobileNetV2: Inverted Residuals and Linear Bottlenecks,''
\textit{CVPR}, pp.~4510--4520, 2018.

\bibitem{mohanty2016}
S.\,P.~Mohanty, D.\,P.~Hughes, and M.~Salath\'{e},
``Using Deep Learning for Image-Based Plant Disease Detection,''
\textit{Front.\ Plant Sci.}, vol.~7, p.~1419, 2016.

\bibitem{selvaraju2017}
R.\,R.~Selvaraju et al., ``Grad-CAM: Visual Explanations from Deep Networks
via Gradient-Based Localization,'' \textit{ICCV}, pp.~618--626, 2017.

\bibitem{fao2023}
FAO, ``FAOSTAT --- Crop and Livestock Products,''
Food and Agriculture Organisation, 2023.

\bibitem{pedregosa2011}
F.~Pedregosa et al., ``Scikit-learn: Machine Learning in Python,''
\textit{JMLR}, vol.~12, pp.~2825--2830, 2011.
\end{thebibliography}

\end{document}
""")

    return "\n".join(parts)


# ── Compile ───────────────────────────────────────────────────────────────────
def compile_pdf(engine):
    for pass_n in range(1, 3):
        print(f"  Compiling pass {pass_n}/2 with {engine}...")
        r = subprocess.run(
            [engine, "-interaction=nonstopmode",
             "-output-directory", str(TEX_DIR), str(TEX_FILE)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            log = (r.stdout + r.stderr).splitlines()
            print("\n  [LaTeX errors — last 50 lines]")
            for ln in log[-50:]:
                print(" ", ln)
            raise RuntimeError(f"{engine} failed")

    compiled = TEX_DIR / "report.pdf"
    if compiled.exists():
        shutil.copy2(str(compiled), str(PDF_OUT))
        size = PDF_OUT.stat().st_size // 1024
        print(f"\n  PDF saved  → {PDF_OUT}  ({size} KB)")
    else:
        print("  [WARN] PDF not found after compilation.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    engine = ensure_latex()
    print("Building LaTeX source...")
    tex = build_tex()
    TEX_FILE.write_text(tex, encoding="utf-8")
    print(f"  .tex written → {TEX_FILE}  ({len(tex)//1024} KB)")
    compile_pdf(engine)


if __name__ == "__main__":
    main()
