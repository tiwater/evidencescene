% evidencescene.cls — Clean Academic Single-Column Preprint Class for EvidenceScene
% Classic serif academic paper layout (Palatino, centered title block, Page-1 teaser,
% unboxed centered Abstract, indented paragraphs, "Figure 1 |" captions).
% Unified with the Tiwater Research preprint house style; compatible with Tectonic (XeTeX) and arXiv pdfLaTeX.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{evidencescene}[2026/11/26 EvidenceScene Academic Single-Column Preprint Class]

\LoadClass[11pt,a4paper]{article}

% --- Page Geometry (A4, generous classic academic margins) ---
\RequirePackage[
  a4paper,
  top=2.45cm,
  bottom=2.55cm,
  left=2.5cm,
  right=2.5cm,
  headheight=18pt,
  headsep=14pt,
  footskip=24pt
]{geometry}

% --- Typography (Classic Academic Palatino + subtle sans & monospace) ---
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[sc]{mathpazo}
\RequirePackage[scaled=0.90]{helvet}
\RequirePackage[scaled=0.84]{inconsolata}
\RequirePackage{microtype}
\linespread{1.08}

% --- Core Packages ---
\RequirePackage[table]{xcolor}
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{booktabs,multirow,tabularx,array}
\RequirePackage{graphicx}
\RequirePackage{enumitem}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{caption}
\RequirePackage{etoolbox}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\RequirePackage[numbers,sort&compress]{natbib}
\setlength{\bibsep}{2.5pt plus 0.5pt}
\RequirePackage{xurl}

% --- Tasteful Restrained Academic Palette (Tiwater House Style) ---
\definecolor{esink}{HTML}{1A1F29}      % Deep charcoal ink
\definecolor{esaccent}{HTML}{1B5E7A}   % Restrained academic slate-teal
\definecolor{esmuted}{HTML}{5F6C7B}    % Metadata / secondary captions
\definecolor{esrule}{HTML}{D0D7DE}     % Delicate hairline rules
\definecolor{esbg}{HTML}{F7F9FB}       % Subtle diagram card fill
\definecolor{esrow}{HTML}{F0F6FA}      % Whisper-light EvidenceScene table row highlight
\definecolor{esgreen}{HTML}{1E6B45}    % Muted verification green
\definecolor{esgreenbg}{HTML}{F2FAF5}
\definecolor{esamber}{HTML}{9A5B13}    % Muted caution amber
\definecolor{esamberbg}{HTML}{FDF8F0}

\RequirePackage[
  colorlinks=true,
  linkcolor=esaccent,
  citecolor=esaccent,
  urlcolor=esaccent,
  pdfborder={0 0 0}
]{hyperref}

% --- Classic Academic Paragraph Style (Indented, balanced spacing) ---
\setlength{\parindent}{1.5em}
\setlength{\parskip}{3pt plus 1pt}
\setlength{\emergencystretch}{2.5em}
\tolerance=1400
\hbadness=2500
\let\es@origunderscore\_
\renewcommand{\_}{\es@origunderscore\allowbreak}

\setlist[itemize]{leftmargin=1.6em, itemsep=2pt, topsep=3pt, parsep=0pt}
\setlist[enumerate]{leftmargin=1.7em, itemsep=2pt, topsep=3pt, parsep=0pt}

% --- Tables & Floats ---
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{6}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.80}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.85}

% --- Classic "Figure 1 | ..." & "Table 1 | ..." Captions ---
\DeclareCaptionLabelSeparator{esbar}{\;\textcolor{esaccent}{\textbf{|}}\;}
\captionsetup{
  font=small,
  labelfont={bf,color=esink},
  textfont={color=esink},
  labelsep=esbar,
  skip=7pt
}
\captionsetup[table]{position=top, skip=6pt}

% --- Classic Academic Serif Section Headings ---
\titleformat{\section}
  {\Large\bfseries\color{esink}}
  {\thesection}{0.75em}{}
\titlespacing*{\section}{0pt}{16pt plus 3pt minus 2pt}{6pt plus 1pt}

\titleformat{\subsection}
  {\large\bfseries\color{esink}}
  {\thesubsection}{0.65em}{}
\titlespacing*{\subsection}{0pt}{12pt plus 2pt minus 1pt}{4pt plus 1pt}

\titleformat{\subsubsection}
  {\normalsize\bfseries\color{esink}}
  {\thesubsubsection}{0.55em}{}
\titlespacing*{\subsubsection}{0pt}{8pt plus 1pt minus 1pt}{2pt}

% --- Delicate Academic Headers & Footers ---
\newcommand{\@esrunningtitle}{EvidenceScene: Observability-Conditioned Pose Tracking and Revisable 3D Scene Reconstruction}
\newcommand{\runningtitle}[1]{\renewcommand{\@esrunningtitle}{#1}}
\newcommand{\@esheaderleft}{Tiwater}
\newcommand{\esheaderleft}[1]{\renewcommand{\@esheaderleft}{#1}}
\newcommand{\@esdate}{November 2026}
\newcommand{\esdate}[1]{\renewcommand{\@esdate}{#1}}

\fancypagestyle{esmain}{
  \fancyhf{}
  \fancyhead[L]{\footnotesize\color{esmuted}\@esrunningtitle}
  \fancyhead[R]{\footnotesize\color{esmuted}\@esdate}
  \fancyfoot[C]{\small\color{esink}\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\headrule}{\hbox to\headwidth{\color{esrule}\leaders\hrule height \headrulewidth\hfill}}
  \renewcommand{\footrulewidth}{0pt}
}

\fancypagestyle{esfirst}{
  \fancyhf{}
  \fancyhead[L]{\footnotesize\sffamily\bfseries\color{esink}EVIDENCESCENE\;\normalfont\color{esrule}|\;\color{esmuted}\@esheaderleft}
  \fancyhead[R]{\footnotesize\sffamily\color{esmuted}\@esdate}
  \fancyfoot[C]{\small\color{esink}\thepage}
  \renewcommand{\headrulewidth}{0.45pt}
  \renewcommand{\headrule}{\hbox to\headwidth{\color{esrule}\leaders\hrule height \headrulewidth\hfill}}
  \renewcommand{\footrulewidth}{0pt}
}

\pagestyle{esmain}

% --- Metadata Macros ---
\newcommand{\@esauthors}{}
\newcommand{\@esaffil}{}
\newcommand{\@eslinks}{}
\newcommand{\esauthors}[1]{\renewcommand{\@esauthors}{#1}}
\newcommand{\esaffil}[1]{\renewcommand{\@esaffil}{#1}}
\newcommand{\eslinks}[1]{\renewcommand{\@eslinks}{#1}}

% --- Authentic Centered Academic Title Block ---
\newcommand{\makeacademictitle}{%
  \thispagestyle{esfirst}%
  \begin{center}
    \vspace*{1pt}
    {\fontsize{14.6}{19}\selectfont\bfseries\color{esink}\@title\par}
    \vspace{8pt}
    {\normalsize\bfseries\color{esink}\@esauthors\par}
    \vspace{2.5pt}
    {\small\color{esmuted}\@esaffil\par}
    \vspace{4pt}
    {\small\sffamily\color{esmuted}\@eslinks\par}
  \end{center}
  \vspace{5pt}
}

% --- Clean Centered Unboxed Academic Abstract ---
\renewenvironment{abstract}{%
  \vspace{2pt}
  {\centering\large\bfseries\color{esink}Abstract\par}
  \vspace{4pt}
  \noindent\ignorespaces
}{%
  \par\vspace{6pt}
}

\endinput
