"""Architecture TikZ — premium design with gradients and shadows."""
import os, sys, subprocess, shutil

PROJ = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(PROJ, "paper", "figures")
TMPDIR = os.path.join(PROJ, "_tikz_tmp")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

TIKZ = r"""\documentclass[border=8pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,shadows.blur,backgrounds,fadings}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}

% Premium colour palette
\definecolor{c1}{HTML}{1A5276}   % deep navy — Input
\definecolor{c1l}{HTML}{D4E6F1}
\definecolor{c2}{HTML}{922B21}   % deep burgundy — D4 Orbit
\definecolor{c2l}{HTML}{FADBD8}
\definecolor{c3}{HTML}{1E8449}   % forest green — MLP
\definecolor{c3l}{HTML}{D5F5E3}
\definecolor{c4}{HTML}{7D3C98}   % deep purple — Reynolds Avg
\definecolor{c4l}{HTML}{E8DAEF}
\definecolor{c5}{HTML}{D35400}   % burnt orange — Output
\definecolor{c5l}{HTML}{FAE5D3}
\definecolor{gdark}{HTML}{2C3E50}
\definecolor{gmid}{HTML}{5D6D7E}
\definecolor{glight}{HTML}{AEB6BF}

\begin{document}
\begin{tikzpicture}[
  scale=1.08, transform shape,
  % Stage panel style
  stage/.style 2 args={
    rectangle, rounded corners=10pt,
    draw=#1!40, line width=1.5pt,
    fill=#2, blur shadow={shadow xshift=1.5pt, shadow yshift=-2pt, shadow blur steps=4, shadow blur radius=2pt},
    minimum width=#1,
  },
  % Section title
  stitle/.style={font=\bfseries\large, text=#1},
  % Body text
  btext/.style={font=\small, text=gdark},
  % Arrow between stages
  arr/.style={-{Stealth[scale=1.6, round]}, line width=2.2pt, color=gmid, rounded corners=6pt},
  % Thin arrow
  tarr/.style={-{Stealth[scale=1.0, round]}, line width=1.2pt, color=glight, rounded corners=3pt},
  % Highlight box
  hlbox/.style={rectangle, rounded corners=5pt, fill=white, draw=#1!30, line width=1.0pt,
                inner sep=6pt, blur shadow={shadow xshift=0.5pt, shadow yshift=-1pt, shadow blur steps=2}},
  % Small label
  slab/.style={font=\fontsize{7}{9}\selectfont, text=gmid},
]

% ===== TITLE =====
\node[font=\huge\bfseries, text=gdark] at (13.5, 15.5)
  {$D_4$\textbf{-PINN}\quad---\quad Reynolds-Averaged Group-Equivariant Architecture};

% ============================================================
% STAGE 1 — INPUT  (x: 0–4.5, y: 3.5–11)
% ============================================================
\draw[rounded corners=10pt, fill=c1l!60, draw=c1!40, line width=1.5pt,
      blur shadow={shadow xshift=1.5pt, shadow yshift=-2pt, shadow blur steps=4}]
  (0.3, 3.5) rectangle (4.8, 11.5);
\node[stitle=c1] at (2.55, 11.95) {\Large Input Space};

% Domain icon
\draw[fill=white, draw=c1!60, line width=1.2pt, rounded corners=3pt] (1.2, 6.0) rectangle ++(1.8,1.8);
% Grid lines
\draw[c1!15, dashed, line width=0.35pt] (2.1,6.9) -- ++(0:1.0) (2.1,6.9) -- ++(180:1.0);
\draw[c1!15, dashed, line width=0.35pt] (2.1,6.9) -- ++(90:1.0) (2.1,6.9) -- ++(270:1.0);
\draw[c1!15, dashed, line width=0.35pt] (2.1,6.9) -- ++(45:1.0) (2.1,6.9) -- ++(225:1.0);
\draw[c1!15, dashed, line width=0.35pt] (2.1,6.9) -- ++(135:1.0) (2.1,6.9) -- ++(315:1.0);
\fill[c1] (2.1,6.9) circle (3.5pt);
\node[btext, c1] at (2.55, 5.5) {$\mathbf{x} = (x,\,y)$};
\node[font=\small, text=c1!70] at (2.55, 8.8) {$\Omega = (-1,1)^2$};
\node[font=\small, text=c1!70] at (2.55, 8.3) {square domain};

% ============================================================
% STAGE 2 — D4 ORBIT  (x: 5.5–11.5, y: 3.5–11.5)
% ============================================================
\draw[rounded corners=10pt, fill=c2l!60, draw=c2!40, line width=1.5pt,
      blur shadow={shadow xshift=1.5pt, shadow yshift=-2pt, shadow blur steps=4}]
  (5.5, 3.5) rectangle (11.5, 11.5);
\node[stitle=c2] at (8.5, 11.95) {\Large $D_4$ Orbit Expansion};

% 8 orbit points around center (8.5, 7.3), radius 2.2
% 0 deg
\fill[c2] (8.50+2.20, 7.30) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50+2.20+0.55, 7.30+0.15) {$(x,y)$};
% 45 deg
\fill[c2] (8.50+1.556, 7.30+1.556) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50+1.556+0.45, 7.30+1.556+0.4) {$(-y,x)$};
% 90 deg
\fill[c2] (8.50, 7.30+2.20) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50, 7.30+2.20+0.45) {$(-x,-y)$};
% 135 deg
\fill[c2] (8.50-1.556, 7.30+1.556) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50-1.556-0.45, 7.30+1.556+0.4) {$(y,-x)$};
% 180 deg
\fill[c2] (8.50-2.20, 7.30) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50-2.20-0.55, 7.30+0.15) {$(-x,y)$};
% 225 deg
\fill[c2] (8.50-1.556, 7.30-1.556) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50-1.556-0.45, 7.30-1.556-0.4) {$(x,-y)$};
% 270 deg
\fill[c2] (8.50, 7.30-2.20) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50, 7.30-2.20-0.45) {$(y,x)$};
% 315 deg
\fill[c2] (8.50+1.556, 7.30-1.556) circle (3.0pt);
\node[font=\fontsize{6.5}{8}\selectfont, text=c2!85] at (8.50+1.556+0.45, 7.30-1.556-0.4) {$(-y,-x)$};

% Radial lines
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50+2.20,7.30);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50+1.556,7.30+1.556);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50,7.30+2.20);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50-1.556,7.30+1.556);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50-2.20,7.30);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50-1.556,7.30-1.556);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50,7.30-2.20);
\draw[c2!15, line width=0.3pt] (8.50,7.30) -- (8.50+1.556,7.30-1.556);

\fill[c2!70] (8.50,7.30) circle (2.2pt);
\node[font=\fontsize{7.5}{9}\selectfont\itshape, text=gmid] at (8.50, 4.6) {8-fold batch $\{g\cdot\mathbf{x} : g\in D_4\}$};
\node[slab] at (8.50, 4.1) {$|D_4| = 8$ elements};

% ============================================================
% STAGE 3 — SHARED MLP  (x: 12.2–17.3, y: 3.5–11.5)
% ============================================================
\draw[rounded corners=10pt, fill=c3l!60, draw=c3!40, line width=1.5pt,
      blur shadow={shadow xshift=1.5pt, shadow yshift=-2pt, shadow blur steps=4}]
  (12.2, 3.5) rectangle (17.5, 11.5);
\node[stitle=c3] at (14.85, 11.95) {\Large Shared MLP $f_\theta$};

% MLP as abstract layered rectangles
% Layer 1: Input (2)
\draw[fill=white, draw=c1!50, line width=1.0pt, rounded corners=3pt] (13.2, 5.8) rectangle ++(0.55,4.6);
\node[font=\fontsize{7}{9}\selectfont, text=c1!80] at (13.475, 5.3) {2};
% Layer 2: Hidden (32)
\draw[fill=white, draw=c3!50, line width=1.0pt, rounded corners=3pt] (14.3, 5.8) rectangle ++(0.55,4.6);
\node[font=\fontsize{7}{9}\selectfont, text=c3!80] at (14.575, 5.3) {32};
% Layer 3: Hidden (32)
\draw[fill=white, draw=c3!50, line width=1.0pt, rounded corners=3pt] (15.4, 5.8) rectangle ++(0.55,4.6);
\node[font=\fontsize{7}{9}\selectfont, text=c3!80] at (15.675, 5.3) {32};
% Layer 4: Output (1)
\draw[fill=white, draw=c4!50, line width=1.0pt, rounded corners=3pt] (16.5, 5.8) rectangle ++(0.55,4.6);
\node[font=\fontsize{7}{9}\selectfont, text=c4!80] at (16.775, 5.3) {1};

% Inter-layer arrows
\draw[tarr] (13.75, 8.1) -- (14.3, 8.1);
\draw[tarr] (14.85, 8.1) -- (15.4, 8.1);
\draw[tarr] (15.95, 8.1) -- (16.5, 8.1);

\node[hlbox=c3!40] at (14.85, 10.5) {Activation: $\sigma = \tanh$};
\node[slab] at (14.85, 4.3) {Batched $\times 8$ (\texttt{einsum})};
\node[slab] at (14.85, 3.85) {1185 parameters};

% ============================================================
% STAGE 4 — REYNOLDS AVERAGE (above MLP)
% ============================================================
\draw[rounded corners=10pt, fill=c4l!70, draw=c4!40, line width=1.5pt,
      blur shadow={shadow xshift=1.5pt, shadow yshift=-2pt, shadow blur steps=4}]
  (12.5, 12.3) rectangle (17.2, 14.5);
\node[stitle=c4] at (14.85, 14.85) {\normalsize Reynolds Average $\mathcal{R}_{D_4}$};

\node[rounded corners=5pt, fill=c4, text=white, font=\bfseries\footnotesize,
      minimum width=4.2cm, minimum height=0.85cm] at (14.85, 13.45)
  {$\displaystyle\frac{1}{|D_4|}\sum_{g\in D_4} f_\theta(g\cdot\mathbf{x})$};

% ============================================================
% STAGE 5 — OUTPUT  (x: 18.2–22.5, y: 3.5–11.5)
% ============================================================
\draw[rounded corners=10pt, fill=c5l!60, draw=c5!40, line width=1.5pt,
      blur shadow={shadow xshift=1.5pt, shadow yshift=-2pt, shadow blur steps=4}]
  (18.2, 3.5) rectangle (22.3, 11.5);
\node[stitle=c5] at (20.25, 11.95) {\Large Output};

\node[rounded corners=6pt, fill=c5, text=white, minimum width=3.2cm, minimum height=4.0cm,
      blur shadow={shadow xshift=1pt, shadow yshift=-1.5pt, shadow blur steps=4}]
  at (20.25, 7.8) {};
\node[font=\huge\bfseries, white] at (20.25, 8.8) {$u_\theta(\mathbf{x})$};
\node[font=\large, white!90] at (20.25, 7.8) {$D_4$-invariant};
\node[font=\large, white!90] at (20.25, 7.1) {solution};
\node[font=\normalsize, c5l!80] at (20.25, 5.8) {\textbf{\LARGE\checkmark}\, Sym.\ $\sim\!10^{-8}$};

% ============================================================
% ARROWS — connecting stages
% ============================================================
% Input -> Orbit
\draw[arr, c2!70] (4.85, 8.5) -- (5.5, 8.5);

% Orbit -> MLP
\draw[arr, c3!70] (11.55, 8.5) -- (12.2, 8.5);

% MLP -> Reynolds (up arrow)
\draw[arr, c4!70] (14.85, 11.5) -- (14.85, 12.3);

% Reynolds -> Output (path: right from Reynolds, down to Output)
\draw[arr, c5!70] (17.25, 13.4) -- (18.2, 13.4) -- (18.2, 8.5);

% ============================================================
% PROPOSITION CALLOUT — across bottom
% ============================================================
\draw[rounded corners=8pt, fill=gdark!3, draw=glight!40, line width=0.9pt]
  (0.3, 1.3) rectangle (22.3, 2.5);
\node[font=\fontsize{8.5}{11}\selectfont, text=gdark!70, text width=21cm, align=center] at (11.3, 1.9)
  {\textbf{Proposition 1 (Architectural Invariance):}\quad
   $\forall\,\theta\in\Theta,\ \forall\,\mathbf{x}\in\mathbb{R}^2,\
   u_\theta(g'\cdot\mathbf{x}) = u_\theta(\mathbf{x})$ for all $g'\in D_4$ \quad
   (the Reynolds average makes every forward pass exactly $D_4$-invariant at machine precision)};

% ============================================================
% LEGEND
% ============================================================
\fill[c1] (0.3, 0.4) rectangle ++(0.4,0.35);
\node[font=\fontsize{7.5}{9}\selectfont, anchor=west, text=gdark] at (0.85, 0.575) {Input Space};
\fill[c2] (4.5, 0.4) rectangle ++(0.4,0.35);
\node[font=\fontsize{7.5}{9}\selectfont, anchor=west, text=gdark] at (5.05, 0.575) {$D_4$ Orbit Expansion};
\fill[c3] (9.2, 0.4) rectangle ++(0.4,0.35);
\node[font=\fontsize{7.5}{9}\selectfont, anchor=west, text=gdark] at (9.75, 0.575) {Shared MLP $f_\theta$};
\fill[c4] (13.5, 0.4) rectangle ++(0.4,0.35);
\node[font=\fontsize{7.5}{9}\selectfont, anchor=west, text=gdark] at (14.05, 0.575) {Reynolds Average};
\fill[c5] (18.2, 0.4) rectangle ++(0.4,0.35);
\node[font=\fontsize{7.5}{9}\selectfont, anchor=west, text=gdark] at (18.75, 0.575) {Output};

\end{tikzpicture}
\end{document}
"""

tex_path = os.path.join(TMPDIR, "fig0_architecture.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(TIKZ)

clean_env = os.environ.copy()
path = clean_env.get("PATH", "")
clean_dirs = [d for d in path.split(os.pathsep) if ".local" not in d]
clean_env["PATH"] = os.pathsep.join(clean_dirs)

for i in range(2):
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", TMPDIR, tex_path],
        capture_output=True, text=True, env=clean_env, timeout=60,
    )
    log_path = os.path.join(TMPDIR, "fig0_architecture.log")
    if os.path.exists(log_path):
        with open(log_path) as lf:
            log = lf.read()
        if "Fatal error" in log:
            print(f"ERROR in pass {i+1}:")
            for line in log.split("\n"):
                if line.startswith("!"):
                    print(f"  {line}")
            break
    elif result.returncode != 0:
        print(f"Pass {i+1} failed with code {result.returncode}")

pdf_src = os.path.join(TMPDIR, "fig0_architecture.pdf")
pdf_dst = os.path.join(OUTDIR, "fig0_architecture.pdf")
if os.path.exists(pdf_src):
    shutil.copy(pdf_src, pdf_dst)
    size_kb = os.path.getsize(pdf_dst) / 1024
    print(f"OK: {size_kb:.0f} KB -> {pdf_dst}")
else:
    print("FAILED")
